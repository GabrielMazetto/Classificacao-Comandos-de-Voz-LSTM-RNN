# 🚀 Classificação de Comandos de Voz com RNN/LSTM

Este projeto implementa e compara diferentes arquiteturas de Redes Neurais Recorrentes (RNN) para a classificação de 20 comandos de voz do dataset "Google Speech Commands".

O foco principal é avaliar como diferentes complexidades de modelos (LSTM e SimpleRNN) lidam com os dados, e determinar a abordagem de pré-processamento mais eficaz (Áudio Bruto vs. MFCCs).

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## 📊 Dataset: Google Speech Commands

* **Fonte:** `tfds.load('speech_commands')`.
* **Formato de Áudio:** Mono, 16000Hz.
* **Classes Alvo:** Foram selecionadas 20 classes de controle (ex: 'down', 'eight', 'five', 'four', 'go', 'left', 'nine', 'no', 'on', 'one', 'right', 'stop', 'up', 'yes', 'zero', etc.).
* **Distribuição:** O dataset foi dividido em:
    * Treino: 37.158 amostras
    * Validação: 5.071 amostras
    * Teste: 5.119 amostras
* **Balanceamento:** O dataset de treino é balanceado, contendo aproximadamente 2.400 arquivos por classe alvo.

## 1. Abordagem Inicial (Descartada): Áudio Bruto

Inicialmente, exploramos o uso do **áudio bruto (waveform)** como entrada.
* **Pré-processamento:** Para contornar o desafio do comprimento variável, aplicamos *downsampling* em cada clipe para **4.000 amostras**.
* **Modelo:** Um modelo LSTM de 3 camadas (LSTM(64) -> LayerNorm -> Dropout -> ...) foi treinado.
* **Parâmetros:** 88.916.
* **Resultado:** Esta abordagem **não se mostrou eficaz**. A acurácia de teste foi de apenas **0.0502** (próximo de um modelo aleatório de 5% para 20 classes). Isso indicou que o áudio bruto *downsampled* não era adequado para a tarefa.

## 2. Abordagem Principal: MFCCs (Mel Frequency Cepstral Coefficients)

Diante do resultado da abordagem inicial, a estratégia seguinte foi utilizar **MFCCs (Coeficientes de Cepstro Frequencial Mel)**.

* **MFCCs** são uma forma de representar o som que foca nas características importantes para a fala humana, como o timbre e o tom, de maneira mais compacta que o áudio bruto. Ao contrário do áudio bruto, que contém muita informação redundante, os MFCCs capturam eficientemente os aspectos relevantes do som, tornando o problema mais tratável.

* **Entrada para o Modelo:** Os MFCCs foram extraídos de forma a gerar uma sequência de **(30, 40)** para cada áudio (30 "quadros" de tempo, cada um descrito por 40 coeficientes). Embora o total seja de 1200 pontos de entrada, esses pontos representam características acústicas relevantes e já processadas, sendo muito mais informativos que as 4000 amostras de áudio bruto da abordagem anterior.

## 🧠 Modelos Avaliados (com MFCCs)

Quatro arquiteturas baseadas em LSTM e uma baseada em SimpleRNN foram treinadas e avaliadas com a entrada de MFCCs (30, 40):

### LSTM Simples
* **Input Shape:** (30, 40)
* **Arquitetura:** Input -> LSTM(64) -> Dense(20, softmax)
* **Parâmetros:** 28.180
* **Acurácia Teste:** 0.8855

### LSTM + Normalização + Dropout
* **Input Shape:** (30, 40)
* **Arquitetura:** Input -> LSTM(64) -> LayerNorm -> Dropout(0.3) -> Dense(20, softmax)
* **Parâmetros:** 28.308
* **Acurácia Teste:** 0.8883

### LSTM + Camada Densa Oculta
* **Input Shape:** (30, 40)
* **Arquitetura:** Input -> LSTM(64) -> LayerNorm -> Dropout(0.3) -> Dense(64, relu) -> LayerNorm -> Dropout(0.3) -> Dense(20, softmax)
* **Parâmetros:** 32.596
* **Acurácia Teste:** 0.8789

### LSTM (3 Camadas) + Camada Densa Oculta (Melhor Modelo)
* **Input Shape:** (30, 40)
* **Arquitetura:** Input -> LSTM(64) -> ... -> LSTM(64) -> ... -> Dense(64, relu) -> ... -> Dense(20, softmax)
* **Parâmetros:** 98.900
* **Acurácia Teste:** **0.9156**
* **Observação:** Obteve o **melhor desempenho**, indicando que a maior complexidade capturou melhor os padrões.

### SimpleRNN (3 Camadas)
* **Input Shape:** (30, 40)
* **Arquitetura:** Input -> SimpleRNN(64) -> ... -> SimpleRNN(64) -> ... -> Dense(20, softmax)
* **Parâmetros:** 24.916
* **Acurácia Teste:** 0.2948
* **Observação:** Desempenho muito inferior, confirmando a dificuldade da RNN simples em lidar com dependências temporais longas.

## 📊 Comparação de Acurácia

O gráfico abaixo resume a acurácia de teste de todos os modelos avaliados sobre os MFCCs.

![Gráfico de comparação de Acurácia dos modelos](/img/image.png)

## 🛠️ Tecnologias Utilizadas

* **Python**
* **TensorFlow** e **Keras:** Para carregar o dataset, pré-processar o áudio e construir/treinar os modelos RNN e LSTM.
* **TensorFlow Datasets (TFDS):** Para o carregamento do dataset `speech_commands`.
* **Librosa:** Para processamento de áudio (incluindo extração de MFCCs).
* **Pandas** e **NumPy:** Para manipulação de dados e cálculos auxiliares.
* **Matplotlib** e **Seaborn:** Para visualização dos dados e resultados.
