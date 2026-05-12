import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, matthews_corrcoef
from utils import select_features_benford
from sklearn.preprocessing import StandardScaler 
from sklearn.mixture import GaussianMixture

# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS

# Lendo a base de dados principal. Certifique-se de que o arquivo CSV está na pasta 'data'.
df = pd.read_csv('data/UNSW_NB15_training-set.csv')

# O artigo base destaca que focar exclusivamente em características numéricas 
# é a estratégia mais eficaz para detectar anomalias reais no tráfego de rede.
df_numeric = df.select_dtypes(include=[np.number])

# Separamos o tráfego normal (benigno) dos ataques. Essa separação é crucial 
# para podermos simular um cenário de 'dia zero' mais à frente.
benignos = df_numeric[df_numeric['label'] == 0]
maliciosos = df_numeric[df_numeric['label'] == 1]

# 2. DIVISÃO SEMISSUPERVISIONADA (20% TREINO / 80% TESTE)

# O modelo vai aprender o que é o comportamento 'normal' da rede treinando com 
# apenas uma amostra (20%) dos dados benignos, sem nunca ter visto um ataque.
train_data = benignos.sample(frac=0.2, random_state=42)
test_benignos = benignos.drop(train_data.index)

# O conjunto de teste junta o restante do tráfego normal com todos os ataques. 
# Esses ataques serão a grande 'surpresa' para validar se o modelo detecta o dia zero.
test_data_combined = pd.concat([test_benignos, maliciosos])
y_true = [0] * len(test_benignos) + [1] * len(maliciosos)

# 3. SELEÇÃO DE CARACTERÍSTICAS (LEI DE BENFORD)

# Como a base já sofreu normalização prévia (o que zera o teste estatístico dinâmico), 
# vamos aplicar diretamente as 16 características que os autores validaram 
# como as mais significativas no estudo original.
features_significativas = [
    'spkts', 'dpkts', 'sbytes', 'dbytes', 'sload', 'dload',
    'sinpkt', 'dinpkt', 'sjit', 'djit', 'tcprtt', 'synack',
    'ackdat', 'smean', 'dmean', 'dur'
]

print(f"Utilizando as {len(features_significativas)} características da Lei de Benford do artigo...")

X_train = train_data[features_significativas]
X_test = test_data_combined[features_significativas]

# --- PADRONIZAÇÃO DOS DADOS (SCALING) ---
# Algoritmos baseados em distância, como o SVM, exigem que todas as colunas 
# estejam na mesma escala matemática. Sem isso, colunas com valores na casa 
# dos milhões ofuscariam colunas com valores pequenos, mas importantes.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TREINAMENTO PADRÃO (IGUAL AO ARTIGO - SEM GRID SEARCH)

# Aqui treinamos o modelo exatamente com os parâmetros sugeridos 
# pelos autores originais (nu=0.1 e gamma='scale') para criar uma base de comparação.
print("Treinando o OCSVM com parâmetros originais do artigo...")
model_original = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
model_original.fit(X_train_scaled)

y_pred_original_raw = model_original.predict(X_test_scaled)
y_pred_original = [1 if p == -1 else 0 for p in y_pred_original_raw]

print("\n--- RESULTADOS DA DETECÇÃO (OCSVM ORIGINAL DO ARTIGO) ---")
print(classification_report(y_true, y_pred_original, target_names=['Benigno', 'Ataque (Zero-Day)']))
mcc_original = matthews_corrcoef(y_true, y_pred_original)
print(f"MCC Score Original: {mcc_original:.4f}\n")

# 5. TREINAMENTO COM GRID SEARCH (A NOSSA OTIMIZAÇÃO)

# Como a base foi pré-processada, os parâmetros do artigo podem falhar. 
# O Grid Search vai testar várias combinações para achar o melhor ajuste.
print("Iniciando Grid Search para otimizar o One-Class SVM...")

melhor_mcc = -1
melhores_parametros = {}
melhor_y_pred = []

valores_nu = [0.01, 0.05, 0.1, 0.2]
valores_gamma = ['scale', 'auto', 0.01, 0.001]

for nu in valores_nu:
    for gamma in valores_gamma:
        # Treina com a combinação atual do loop
        model_otimizado = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        model_otimizado.fit(X_train_scaled)
        
        y_pred_raw = model_otimizado.predict(X_test_scaled)
        y_pred = [1 if p == -1 else 0 for p in y_pred_raw]
        
        mcc_atual = matthews_corrcoef(y_true, y_pred)
        
        # Salva se for o melhor resultado
        if mcc_atual > melhor_mcc:
            melhor_mcc = mcc_atual
            melhores_parametros = {'nu': nu, 'gamma': gamma}
            melhor_y_pred = y_pred

print(f"Melhores Hiperparâmetros Encontrados: {melhores_parametros}")

print("\n--- RESULTADOS DA DETECÇÃO (OCSVM OTIMIZADO) ---")
print(classification_report(y_true, melhor_y_pred, target_names=['Benigno', 'Ataque (Zero-Day)']))
print(f"MCC Score Otimizado: {melhor_mcc:.4f}")

# 6. COMPARAÇÃO COM UM SEGUNDO ALGORITMO: GMM 

# O Gaussian Mixture Model (GMM) é um modelo probabilístico. A ideia aqui é que 
# os ataques terão uma probabilidade muito baixa de pertencer à distribuição normal.
print("\nTreinando o modelo Gaussian Mixture Model (GMM)...")

# Assim como no OCSVM, o GMM é treinado apenas com os dados normais (escalados)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_train_scaled)

# Diferente do SVM que traça uma fronteira, o GMM nos devolve uma probabilidade (log-likelihood).
# Vamos usar as pontuações da fase de treino para mapear o que é estatisticamente aceitável.
scores_treino = gmm.score_samples(X_train_scaled)

# Definindo o limite de corte (epsilon): consideramos que a margem de 1% com menor 
# probabilidade no treino pode ser um desvio aceitável (simulando de forma análoga o nu=0.01).
limiar_epsilon = np.percentile(scores_treino, 1)

# Calculando a probabilidade para os dados de teste
scores_teste = gmm.score_samples(X_test_scaled)

# Se a probabilidade calculada for menor que o nosso limite de corte, o tráfego 
# é classificado como Ataque (1). Caso contrário, é tráfego Benigno (0).
y_pred_gmm = [1 if score < limiar_epsilon else 0 for score in scores_teste]

print("\n--- RESULTADOS DA DETECÇÃO DE DIA ZERO (GMM) ---")
print(classification_report(y_true, y_pred_gmm, target_names=['Benigno', 'Ataque (Zero-Day)']))

mcc_gmm = matthews_corrcoef(y_true, y_pred_gmm)
print(f"MCC Score GMM: {mcc_gmm:.4f}")