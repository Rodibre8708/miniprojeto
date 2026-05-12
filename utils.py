import numpy as np
import pandas as pd
from scipy.stats import chisquare

def get_benford_probabilities():
    """
    Calcula a probabilidade teórica de um número começar com o dígito 1, 2, 3... até 9.
    A Lei de Benford diz que, em sistemas naturais, o dígito 1 aparece mais frequentemente (cerca de 30%), 
    enquanto o 9 aparece menos. Esta função serve para criar essa "régua" de proporções ideais.
    """
    return [np.log10(1 + 1/d) for d in range(1, 10)]

def test_benford_compliance(data):
    # A matemática da Lei de Benford só se aplica a números estritamente positivos,
    # então o primeiro passo é filtrar zeros e números negativos.
    data = data[data > 0]
    
    # Precisamos de um volume mínimo de dados (100 linhas, por exemplo) para que 
    # o teste estatístico seja confiável e não dê falsos positivos.
    if len(data) < 100: 
        return False
        
    # O "pulo do gato": convertemos os números para texto, pegamos a primeira 
    # letra (que representa o 1º dígito do número) e voltamos para inteiro.
    first_digits = data.astype(str).str[0].astype(int)
    first_digits = first_digits[first_digits != 0]
    
    n_samples = len(first_digits)
    if n_samples == 0: return False

    # 1. Observado: Aqui contamos de fato quantas vezes cada dígito de 1 a 9 apareceu nos nossos dados.
    counts = first_digits.value_counts().sort_index()
    observed = [counts.get(d, 0) for d in range(1, 10)]
    
    # 2. Esperado: Aqui pegamos as proporções ideais da Lei de Benford e 
    # multiplicamos pelo total de amostras para saber o que deveríamos encontrar na teoria.
    probs = [np.log10(1 + 1/d) for d in range(1, 10)]
    expected = [p * n_samples for p in probs]
    
    # Teste de Qui-Quadrado: Funciona como o "juiz" para ver se a diferença entre 
    # o que observamos na prática e o que esperávamos na teoria é aceitável.
    # Se o p-valor for menor que 5% (0.05), a diferença é grande demais e a lei foi quebrada.
    _, p_value = chisquare(observed, f_exp=expected)
    return p_value >= 0.05

def select_features_benford(df_benigno, df_malicioso):
    """
    Esta é a inteligência principal extraída do artigo:
    Uma característica da rede só é valiosa (significativa) se ela tiver um comportamento 'natural' 
    durante o uso comum e se tornar 'anômala' (quebrando a lei) durante um ataque.
    """
    significant_features = []
    
    # O foco metodológico do artigo é puramente em características numéricas (como duração do fluxo).
    numeric_cols = df_benigno.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Ignoramos rótulos de identificação e portas de rede (sport, dport)
        # porque o artigo comprova que elas não ajudam a prever ataques de dia zero.
        if col in ['label', 'id', 'attack_cat', 'sport', 'dport']:
            continue
            
        # Realizamos os testes da Lei de Benford nos dois cenários separados.
        follows_benigno = test_benford_compliance(df_benigno[col])
        violates_malicioso = not test_benford_compliance(df_malicioso[col])
        
        # A Regra de Ouro: A coluna obedece à lei no tráfego normal E quebra a lei sob ataque?
        # Se sim, achamos uma característica excelente para treinar o nosso modelo!
        if follows_benigno and violates_malicioso:
            significant_features.append(col)
            
    return significant_features