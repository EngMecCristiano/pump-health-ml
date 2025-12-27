# ⚙️ Pump Health & Reliability ML System

**Um sistema de Inteligência Artificial Industrial de ponta a ponta, projetado para simular, analisar e prever falhas em Bombas Centrífugas.**

Este projeto une a Engenharia de Confiabilidade tradicional (Weibull, RCM) com as mais modernas técnicas de Machine Learning, criando uma solução completa para a gestão de ativos industriais.

---

## 🎯 Visão Geral do Projeto

Esta aplicação funciona como um **"Gêmeo Digital"** (Digital Twin) e uma suíte analítica para equipamentos rotativos. O principal desafio em projetos de manutenção preditiva é a obtenção de dados de falha de alta qualidade. Este projeto aborda essa questão através da geração de datasets sintéticos, informados por princípios da física, e da aplicação de análises de sobrevivência avançadas para extrair insights valiosos.

## ✨ Principais Funcionalidades

| Funcionalidade | Descrição |
| :--- | :--- |
| **Geração de Dados Sintéticos** | Simulação vetorizada de milhares de ativos em milissegundos, utilizando a Decomposição de Cholesky para correlacionar variáveis de sensores (Vibração, Temperatura, Pressão) e modelos de degradação exponencial para simular o desgaste real (Curva da Banheira com Weibull β > 1). |
| **Análise de Sobrevivência** | Estimativas de confiabilidade da frota com curvas de Kaplan-Meier, análise de risco acumulado com estimadores de Nelson-Aalen e modelos de regressão (Cox Proportional Hazards e Weibull AFT) para entender o impacto das covariáveis na vida útil dos componentes. |
| **Modelagem Preditiva (Machine Learning)** | Modelos de classificação para prever o componente que irá falhar (Rolamento, Selo, Rotor) e modelos de regressão para estimar o Tempo de Vida Útil Remanescente (RUL - Remaining Useful Life). |
| **Business Intelligence** | Cálculo automatizado de KPIs essenciais (MTBF, Disponibilidade, Taxa de Censura), geração de Matriz de Risco (Probabilidade vs. Criticidade) e relatórios automatizados em PDF para suportar a tomada de decisão. |

## 🧰 Tecnologias Utilizadas

- **Core:** Python 3.x
- **Interface & Web Framework:** Streamlit
- **Manipulação de Dados:** Pandas, NumPy (com forte vetorização)
- **Confiabilidade e Estatística:** Lifelines, SciPy.stats
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualização de Dados:** Plotly (Interativo), Matplotlib, Seaborn

## 🚀 Instalação e Uso

### Pré-requisitos

- Python 3.8 ou superior
- Git

### Configuração Local

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/EngMecCristiano/pump-health-ml.git
   cd pump-health-ml
   ```

2. **Crie e ative um ambiente virtual (recomendado):**
   ```bash
   # Usando venv
   python3 -m venv .venv
   source .venv/bin/activate

   # Ou usando uv (mais rápido)
   uv venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Lógica de Engenharia e Matemática

Este projeto vai além da simples visualização de dados. O gerador de dados implementa:

- **Distribuição de Weibull:** $f(t) = \frac{\beta}{\eta}(\frac{t}{\eta})^{\beta-1}e^{-(t/\eta)^\beta}$ para modelar a vida base dos componentes.
- **Correlação Multivariada:** Os dados dos sensores não são aleatórios. Uma matriz de covariância garante que, se a vibração aumentar, a temperatura provavelmente seguirá o mesmo padrão, simulando o acoplamento mecânico real.
- **Censura:** Simula dados de manutenção do mundo real, onde nem todos os ativos falharam (Dados Censurados à Direita).

## 👨‍💻 Sobre o Autor

**Cristiano Sacramento**

Engenheiro Mecânico Sênior | Especialista em Confiabilidade | Entusiasta de Data Science

Com mais de 15 anos de experiência na indústria pesada (Mineração, Óleo & Gás), sou especialista em traduzir o comportamento de ativos físicos em insights de dados acionáveis.

- **LinkedIn:** [https://www.linkedin.com/in/cristiano-sacramento-a53a8138/](https://www.linkedin.com/in/cristiano-sacramento-a53a8138/)

Este projeto é para fins educacionais e de portfólio, demonstrando a aplicação de Python na Engenharia de Confiabilidade.
