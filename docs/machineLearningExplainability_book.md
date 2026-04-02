# 📘 Machine Learning Explainability — Guia Prático e Comentado

<br>

> ## *Sumário Oficial*
<details>

### **Capítulo 1 — Introdução à Explainability**
<details>

- 1.1 Por que interpretar modelos  
- 1.2 O problema da “caixa‑preta”  
- 1.3 Tipos de explicabilidade  
- 1.4 Como o Kaggle aborda o tema  
- 1.5 Métricas e modelos usados no curso  
- 1.6 Como este livro está organizado  
</details>

---

### **Capítulo 2 — Permutation Importance**
<details>

- 2.1 O que é importância de features  
- 2.2 Por que permutar valores funciona  
- 2.3 Como interpretar quedas de desempenho  
- 2.4 Exemplo conceitual com Random Forest  
- 2.5 Comparação com Feature Importances tradicionais  
- 2.6 Erros comuns  
- 2.7 Boas práticas  
- 2.8 Checklist  
- 2.9 Glossário  
</details>

---

### **Capítulo 3 — Partial Dependence Plots (PDPs)**
<details>

- 3.1 O que são PDPs  
- 3.2 Intuição visual  
- 3.3 PDPs univariados  
- 3.4 PDPs bivariados  
- 3.5 Como interpretar curvas  
- 3.6 Limitações e armadilhas  
- 3.7 Boas práticas  
- 3.8 Checklist  
- 3.9 Glossário  
</details>

---

### **Capítulo 4 — SHAP Values**
<details>

- 4.1 O que são valores SHAP  
- 4.2 Intuição do método de Shapley  
- 4.3 SHAP para árvores (TreeSHAP)  
- 4.4 SHAP Summary Plots  
- 4.5 SHAP Dependence Plots  
- 4.6 SHAP Force Plots  
- 4.7 Interpretação prática  
- 4.8 Erros comuns  
- 4.9 Boas práticas  
- 4.10 Checklist  
- 4.11 Glossário  
</details>

---

### **Capítulo 5 — Explainability Avançada com SHAP**
<details>

- 5.1 Interações entre features  
- 5.2 Efeitos não lineares  
- 5.3 SHAP Interaction Values  
- 5.4 Casos reais de uso  
- 5.5 Como comunicar resultados  
- 5.6 Checklist  
- 5.7 Glossário  
</details>

---

### **Capítulo 6 — Estudo de Caso Completo**
<details>

- 6.1 Problema real  
- 6.2 Construção do modelo  
- 6.3 Aplicação de Permutation Importance  
- 6.4 Aplicação de PDPs  
- 6.5 Aplicação de SHAP  
- 6.6 Conclusões e recomendações  
- 6.7 Checklist final  
</details>

---

### **Capítulo 7 — Conclusão Geral**
<details>

- 7.1 O que aprendemos  
- 7.2 Como aplicar em projetos reais  
- 7.3 Como comunicar explicabilidade  
- 7.4 Próximos passos na trilha de ML  
</details>
</details>

<br>

---

<br>

# 📘 Capítulo 1 — Introdução à Explainability  
<details>

### *Machine Learning Explainability — Um Guia Prático e Comentado*

<br>

---

## 🟦 1.1. Introdução
<details>
<br>

Modelos modernos de Machine Learning são extremamente poderosos. Eles conseguem prever preços, detectar fraudes, recomendar produtos e até auxiliar diagnósticos médicos.  
Mas existe um problema fundamental:

**Muitos modelos funcionam como “caixas‑pretas”: fazem previsões excelentes, mas não explicam como chegaram lá.**

O objetivo deste livro — que acompanha o curso *Kaggle Machine Learning Explainability* — é justamente abrir essa caixa‑preta.  
Não para revelar fórmulas internas, mas para entender **como o modelo usa os dados para tomar decisões**.

Neste capítulo, você vai explorar:

- por que precisamos de explicabilidade;  
- quando ela é necessária;  
- quais tipos de insights são possíveis;  
- como esses insights ajudam em projetos reais;  
- como o dataset sintético do curso foi construído.

</details>

---

## 🟩 1.2. Revisão do fluxo anterior
<details>
<br>

No livro anterior (*Intermediate Machine Learning*), você aprendeu a:

- tratar missing values;  
- lidar com variáveis categóricas;  
- construir pipelines;  
- validar modelos;  
- usar XGBoost;  
- evitar data leakage.

Ou seja: aprendeu a **construir modelos robustos**.

Agora, o foco muda:

**Em vez de construir modelos, vamos aprender a interpretá‑los.**

E, para isso, utilizaremos um **dataset sintético**, criado especialmente para este curso, já que o dataset original do Kaggle Learn não está mais disponível.

</details>

---

## 🟧 1.3. Apresentação do problema
<details>
<br>

A maioria dos modelos modernos — Random Forests, Gradient Boosting, XGBoost, redes neurais — não fornece explicações claras sobre:

- quais features são mais importantes;  
- como cada feature influencia uma previsão específica;  
- como o modelo se comporta em diferentes regiões do espaço de dados.

Sem essas respostas, enfrentamos riscos como:

- confiar em padrões espúrios;  
- não perceber bugs ou data leakage;  
- tomar decisões erradas;  
- perder oportunidades de melhorar o modelo;  
- não conseguir justificar previsões para stakeholders.

Explainability resolve exatamente isso.

<br>

> ### 🔎 Nota sobre o dataset utilizado no curso  
>  
> O dataset original da Lesson 1 do Kaggle Learn não está mais disponível para download.  
> Para manter a fidelidade pedagógica do curso, utilizamos um **dataset sintético** que reproduz:
>
> - a estrutura das variáveis originais;  
> - o tipo de problema (readmissão hospitalar);  
> - a mistura de variáveis numéricas e categóricas;  
> - e a lógica de geração do target.  
>
> Esse dataset não representa pacientes reais, mas é suficientemente realista para demonstrar todas as técnicas de explicabilidade apresentadas ao longo do curso.

</details>

---

## 🟨 1.4. Conceito central — Por que precisamos de insights?
<details>
<br>

A Lesson 1 do Kaggle apresenta **cinco grandes motivos** para extrair insights de modelos:

### **1) Debugging**
Modelos podem aprender padrões errados por causa de:

- dados sujos;  
- pré‑processamento incorreto;  
- target leakage;  
- correlações espúrias.

Entender o que o modelo está “vendo” é a forma mais rápida de detectar erros.

---

### **2) Feature Engineering**
Criar boas features é a forma mais eficiente de melhorar modelos.

Insights ajudam a:

- identificar features importantes;  
- detectar interações;  
- sugerir transformações úteis.

---

### **3) Direcionar coleta de dados**
Empresas podem decidir:

- quais dados coletar;  
- quais sensores instalar;  
- quais campos adicionar em formulários.

Explainability mostra **quais features realmente agregam valor**, guiando investimentos.

---

### **4) Apoiar decisões humanas**
Nem toda decisão é automatizada.

Em áreas como:

- medicina;  
- crédito;  
- políticas públicas;

insights são mais valiosos que previsões.

---

### **5) Construir confiança**
Stakeholders precisam confiar no modelo.

Mostrar que:

- as features importantes fazem sentido;  
- o modelo não está usando informações indevidas;  
- as relações aprendidas são coerentes;

é essencial para adoção.

<br>

> Embora o dataset utilizado seja sintético, os problemas apresentados aqui são reais:  
> modelos podem aprender padrões errados, ignorar variáveis importantes ou se apoiar em correlações espúrias.  
> A explicabilidade ajuda a revelar esses comportamentos.

</details>

---

## 🟦 1.5. Exemplos conceituais com código
<details>
<br>

A Lesson 1 do Kaggle Learn não incluía código originalmente, mas nesta versão do curso introduzimos um pequeno exemplo prático usando o **dataset sintético** criado no notebook.

O objetivo não é treinar um modelo complexo, mas ilustrar o conceito de *insights de modelo*.

~~~python
from sklearn.ensemble import RandomForestClassifier

# Exemplo ilustrativo usando o dataset sintético
model = RandomForestClassifier(random_state=0)
model.fit(X_train_enc, y_train)

# Importância tradicional de features
importances = model.feature_importances_
~~~

### ✔ O que esse código faz
- Treina um modelo simples  
- Extrai importâncias internas das árvores  

### ✔ Por que isso importa
- É o insight mais básico sobre o comportamento do modelo  
- Mas é limitado — por isso veremos técnicas melhores:
  - Permutation Importance  
  - Partial Dependence Plots  
  - SHAP Values  

</details>

---

## 🟫 1.6. Integração com capítulos posteriores
<details>
<br>

Este capítulo estabelece a motivação.  
Nos próximos capítulos, veremos técnicas específicas para extrair insights:

- **Permutation Importance** — importância global de features  
- **PDPs** — efeito médio de uma feature  
- **SHAP Values** — explicações locais e globais  
- **SHAP avançado** — interações e efeitos não lineares  

Cada técnica responde a uma pergunta diferente sobre o modelo.

</details>

---

## 🟪 1.7. Boas práticas e limitações
<details>
<br>

### ✔ Boas práticas
- Validar se os insights fazem sentido no mundo real  
- Usar múltiplas técnicas para obter visão completa  
- Tratar insights como hipóteses, não verdades absolutas  
- Documentar descobertas para orientar o time  
- Em datasets sintéticos, usar insights como ferramenta pedagógica  

---

### ⚠ Limitações
- Insights não corrigem modelos ruins  
- Explicabilidade não substitui validação  
- Técnicas diferentes podem gerar interpretações diferentes  
- Interpretabilidade não elimina vieses — apenas os revela  

</details>

---

## 📘 1.8. Glossário técnico
<details>
<br>

- **Explainability** — técnicas para interpretar modelos  
- **Insight** — informação sobre como o modelo usa os dados  
- **Feature Importance** — medida de relevância de uma feature  
- **Debugging** — processo de identificar erros no modelo  
- **Feature Engineering** — criação de novas features  
- **Stakeholder** — pessoa que depende das previsões do modelo  

</details>

---

## 🧾 1.9. Referência rápida
<details>
<br>

- Modelos são caixas‑pretas por padrão  
- Explainability revela como o modelo usa os dados  
- Insights ajudam em debugging, feature engineering e confiança  
- O curso ensinará técnicas práticas para extrair esses insights  
- O dataset sintético permite reproduzir o fluxo do Kaggle Learn  

</details>

---

## 🟧 1.10. Conclusão do capítulo
<details>
<br>

Este capítulo apresentou a motivação para estudar explicabilidade e introduziu o dataset sintético que será usado ao longo do curso.  
Modelos poderosos não bastam — precisamos entender **como** eles tomam decisões.

Nos próximos capítulos, aplicaremos técnicas práticas para extrair insights desse modelo, começando por:

> **Permutation Importance** — a primeira técnica prática de explicabilidade.

</details>
</details>
<br>

---

<br>

# 📘 Capítulo 2 — Permutation Importance  
<details>

### *Machine Learning Explainability — Um Guia Prático e Comentado*

<br>

---

## 🟦 2.1. Introdução
<details>
<br>

Uma das perguntas mais fundamentais em interpretabilidade é:

> **Quais features têm maior impacto nas previsões do modelo?**

Essa pergunta parece simples, mas a resposta depende de **como** medimos importância.  
Modelos complexos — como Random Forests, Gradient Boosting e redes neurais — não fornecem explicações diretas sobre o papel de cada variável.

A Lesson 2 do Kaggle apresenta uma técnica poderosa, simples e amplamente utilizada:

> **Permutation Importance**

Ela mede o quanto o modelo depende de cada feature observando **como a performance se deteriora** quando embaralhamos uma coluna do conjunto de validação.

Este capítulo explica:

- por que essa técnica existe;  
- como ela funciona;  
- o que ela mede (e o que não mede);  
- como interpretá-la corretamente;  
- como ela prepara o terreno para técnicas mais avançadas.

</details>

---

## 🟩 2.2. Revisão do fluxo anterior
<details>
<br>

No capítulo anterior, discutimos:

- por que precisamos de explicabilidade;  
- como insights ajudam debugging, feature engineering e confiança;  
- o dataset sintético que usaremos ao longo do curso;  
- a motivação para entender o comportamento interno de modelos.

Agora avançamos para a **primeira técnica prática** de interpretabilidade global:  
**Permutation Importance**, que mede o impacto de cada feature no desempenho do modelo.

</details>

---

## 🟧 2.3. Apresentação do problema
<details>
<br>

Imagine que você treinou um modelo e ele apresenta boa performance.  
Mas surge a pergunta:

> **O modelo está usando as features certas?**

Sem essa resposta, você corre riscos como:

- confiar em padrões espúrios;  
- não perceber data leakage;  
- superestimar a importância de variáveis irrelevantes;  
- subestimar variáveis realmente úteis;  
- tomar decisões erradas sobre coleta de dados.

A Lesson 2 do Kaggle ilustra isso com um exemplo simples:  
prever a altura de uma pessoa aos 20 anos usando informações disponíveis aos 10 anos.

Algumas features são úteis (altura aos 10 anos).  
Outras são irrelevantes (quantidade de meias).

Permutation Importance permite medir isso **sem alterar o modelo**.

</details>

---

## 🟨 2.4. Conceito central — Como funciona o Permutation Importance
<details>
<br>

A ideia central é elegante:

> **Se embaralhar uma coluna piora muito a performance, então o modelo dependia fortemente dessa coluna.**

O processo é:

1. Treine o modelo normalmente.  
2. Pegue o conjunto de validação.  
3. Escolha uma coluna e **embaralhe seus valores**.  
4. Meça a performance do modelo com essa coluna embaralhada.  
5. A queda de performance = **importância da feature**.  
6. Desfaça o embaralhamento.  
7. Repita para todas as colunas.

💡 A técnica funciona para qualquer modelo: lineares, árvores, boosting, redes neurais, etc.


<br>

### 🔍 Por que isso funciona?

Porque embaralhar uma coluna:

- remove qualquer relação entre ela e o target;  
- preserva a distribuição dos valores;  
- mantém todas as outras features intactas.

Se o modelo dependia daquela coluna, suas previsões pioram.  
Se não dependia, nada muda.

<br>

### 🔁 Repetições e variabilidade

A técnica envolve aleatoriedade.  
Por isso, repetimos o embaralhamento várias vezes e calculamos:

- média da queda de performance  
- desvio padrão da queda  

Isso explica por que os resultados aparecem como:

```
0.1750 ± 0.0848
```

📌 *Observação importante:*  
A importância calculada depende da métrica usada.  
Em regressão, medimos aumento do erro (MAE, RMSE).  
Em classificação, medimos queda de acurácia, F1, etc.


</details>

---

## 🟦 2.5. Exemplos conceituais com código
<details>
<br>

A seguir, exemplos **conceituais**, inspirados na Lesson 2.  
O objetivo é ilustrar a técnica, não reproduzir o notebook.

---

### 🧩 Exemplo 1 — Permutation Importance manual (versão simples)

~~~python
from sklearn.metrics import accuracy_score
import numpy as np

baseline = accuracy_score(y_valid, model.predict(X_valid))

def perm_importance(model, X_valid, y_valid, col):
    X_temp = X_valid.copy()
    X_temp[col] = np.random.permutation(X_temp[col])
    shuffled_pred = model.predict(X_temp)
    return baseline - accuracy_score(y_valid, shuffled_pred)

print("Importância de age:", perm_importance(model, X_valid, y_valid, "age"))
~~~

**O que esse código faz:**  
- Calcula a performance original  
- Embaralha uma coluna  
- Mede a queda de performance  

---

### 🧩 Exemplo 2 — Permutation Importance com repetição (mais robusto)

~~~python
def perm_importance_repeated(model, X_valid, y_valid, col, n=10):
    drops = []
    for _ in range(n):
        drops.append(perm_importance(model, X_valid, y_valid, col))
    return np.mean(drops), np.std(drops)

mean, std = perm_importance_repeated(model, X_valid, y_valid, "age")
print(f"Importância de age: {mean:.4f} ± {std:.4f}")
~~~

**Por que isso importa:**  
- Reduz variabilidade  
- Aproxima o comportamento da biblioteca `eli5`  

---

### 🧩 Exemplo 3 — Permutation Importance com eli5 (como no Kaggle)

~~~python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names=X_valid.columns.tolist())
~~~

### Exemplo de saída típica de Permutation Importance

Feature                | Importance
---------------------- | -------------------------
abs_lat_change         | 0.175 ± 0.08
abs_lon_change         | 0.120 ± 0.05
pickup_latitude        | 0.045 ± 0.02
dropoff_latitude       | 0.040 ± 0.02
pickup_longitude       | 0.010 ± 0.01


**Por que usar eli5:**  
- Implementação otimizada  
- Interface clara  
- Resultados reprodutíveis  

---

### 🧩 Exemplo 4 — Comparando duas features correlacionadas

~~~python
for col in ["age", "age_squared"]:
    print(col, perm_importance(model, X_valid, y_valid, col))
~~~

**O que isso demonstra:**  
- Features correlacionadas podem “dividir” importância  
- Permutation Importance não detecta redundância estrutural  


### Exemplo adicional — Features correlacionadas dividem importância

Se `age` e `age_squared` carregam a mesma informação, o modelo pode dividir a importância entre elas, mesmo que uma delas seja redundante.


---

### 🧩 Exemplo 5 — Permutation Importance em regressão

~~~python
from sklearn.metrics import mean_absolute_error

baseline = mean_absolute_error(y_valid, model.predict(X_valid))

def perm_importance_reg(model, X_valid, y_valid, col):
    X_temp = X_valid.copy()
    X_temp[col] = np.random.permutation(X_temp[col])
    return mean_absolute_error(y_valid, model.predict(X_temp)) - baseline
~~~

**Por que isso importa:**  
- A técnica funciona para qualquer métrica  
- A interpretação muda: agora medimos **aumento do erro**  

</details>

---

## 🟫 2.6. Integração com capítulos anteriores
<details>
<br>

Permutation Importance conecta-se diretamente ao Capítulo 1:

- Ele fornece **insights globais** sobre o modelo.  
- Ajuda a identificar **features realmente relevantes**.  
- Permite detectar **padrões espúrios**.  
- Serve como base para técnicas mais avançadas, como:
  - Partial Dependence Plots (PDPs)  
  - SHAP Values  

É a primeira ferramenta prática para “abrir a caixa‑preta”.

📌 Comparação rápida:  
A importância nativa de árvores mede redução de impureza, mas pode ser enviesada.  
Permutation Importance mede impacto real na performance e é mais confiável.


</details>

---

## 🟪 2.7. Boas práticas e limitações
<details>
<br>

### ✔ Boas práticas
- Usar sempre um conjunto de **validação**, nunca o de treino.  
- Repetir embaralhamentos para reduzir variabilidade.  
- Comparar importâncias relativas, não absolutas.  
- Usar como ferramenta de **debugging** e **sanidade**.  
- Verificar se features importantes fazem sentido no domínio.
- Em datasets pequenos, a técnica pode ser dominada pelo ruído, produzindo importâncias instáveis.
- Em modelos muito grandes, o custo computacional cresce proporcionalmente ao número de features × número de repetições.


---

### ⚠ Limitações
- Não funciona bem com features altamente correlacionadas.  
- Pode subestimar a importância de variáveis redundantes.  
- Depende da métrica de performance escolhida.  
- Pode ser instável em datasets pequenos.  
- Não revela **como** a feature afeta a previsão — apenas **se** afeta.

</details>

---

## 📘 2.8. Glossário técnico
<details>
<br>

- **Permutation Importance** — técnica que mede a queda de performance ao embaralhar uma feature.  
- **Importância global** — impacto médio de uma feature no modelo.  
- **Embaralhamento** — permutação aleatória dos valores de uma coluna.  
- **Baseline** — performance original do modelo antes do embaralhamento.  
- **Variabilidade** — flutuação natural causada pela aleatoriedade do processo.  
- **Redundância** — quando duas features carregam a mesma informação.

</details>

---

## 🧾 2.9. Referência rápida
<details>
<br>

- Permutation Importance mede **dependência do modelo** em cada feature.  
- Quanto maior a queda de performance, maior a importância.  
- Técnica simples, rápida e amplamente utilizada.  
- Ideal para debugging e validação de modelos.  
- Prepara o terreno para PDPs e SHAP.  

</details>

---

## 🟧 2.10. Conclusão do capítulo
<details>
<br>

Permutation Importance é a porta de entrada para interpretabilidade prática.  
Ela responde à pergunta fundamental:

> **Quais features realmente importam para o meu modelo?**

Nos próximos capítulos, aprofundaremos a análise com técnicas que mostram:

- **como** cada feature afeta as previsões (PDPs),  
- **por que** uma previsão específica foi feita (SHAP),  
- **como** features interagem entre si.

O próximo capítulo apresenta:

> **Partial Dependence Plots — visualizando o efeito de uma feature nas previsões.**

</details>
</details>
<br>

---

<br>

# 📘 Capítulo 3 — Partial Dependence Plots (PDPs)
<details>

### *Machine Learning Explainability — Um Guia Prático e Comentado*

<br>

---

## 🟦 3.1. Introdução
<details>
<br>

No capítulo anterior, aprendemos a responder uma pergunta essencial:

**Quais features mais influenciam o modelo?**

Mas isso não basta.  
Saber *o que importa* não revela *como importa*.

Para isso, precisamos de uma técnica capaz de isolar o efeito de uma variável, mantendo todas as outras constantes.  
Essa técnica é o **Partial Dependence Plot (PDP)**.

Ele permite visualizar:

- a direção do efeito (positivo, negativo, não linear);
- a intensidade do efeito;
- regiões onde o modelo é mais sensível;
- possíveis interações entre variáveis.

Este capítulo explica a intuição, o funcionamento e as limitações dos PDPs, preparando você para os exercícios da Lesson 3 do Kaggle.

</details>

---

## 🟩 3.2. Revisão do fluxo anterior
<details>
<br>

No Capítulo 2 vimos:

- Permutation Importance → *o que importa*  
- Mas não → *como cada feature afeta a previsão*

Agora avançamos para a primeira técnica que revela **a forma da relação** entre feature e previsão.

</details>

---

## 🟧 3.3. Apresentação do problema
<details>
<br>

Mesmo sabendo quais variáveis são importantes, ainda falta responder:

**Se eu variar apenas uma feature, mantendo todas as outras iguais, como a previsão muda?**

Essa pergunta é essencial para:

- validar comportamento do modelo;
- identificar efeitos não lineares;
- investigar interações;
- comunicar resultados para stakeholders;
- detectar padrões espúrios.

Sem uma ferramenta adequada, seria impossível separar o efeito de uma feature das demais.

É aqui que entram os **Partial Dependence Plots**.

</details>

---

## 🟨 3.4. Conceito central — O que é um Partial Dependence Plot
<details>
<br>

A ideia central é simples:

> PDP mostra como a previsão média muda quando variamos uma feature, mantendo todas as outras constantes.

### 🔍 Intuição fundamental

Pegue uma linha do dataset:

- idade = 40  
- renda = 8.000  
- score = 720  

Agora faça um experimento mental:

1. Mantenha tudo igual.  
2. Varie apenas a idade (20, 30, 40, 50…).  
3. Observe como a previsão muda.  
4. Repita para várias linhas.  
5. Tire a média.

O resultado é uma curva que mostra:

- se a relação é linear;
- se há saturação;
- se há regiões de instabilidade;
- se o modelo aprendeu padrões estranhos.

### 🔬 O que o PDP mede

- efeito marginal médio de uma feature;
- comportamento global do modelo;
- forma da relação (reta, curva, U, degraus).

### 🚫 O que o PDP não mede

- efeitos individuais (isso é papel do ICE e SHAP);
- causalidade;
- efeitos condicionais;
- comportamento em regiões sem dados.

</details>

---

## 🟦 3.5. Exemplos conceituais com código
<details>
<br>

A seguir, exemplos **conceituais**, não ligados ao dataset do Kaggle.

---

### 🧩 Exemplo 1 — PDP manual (intuição)

~~~python
import numpy as np

def pdp_manual(model, X, feature, grid):
    preds = []
    X_temp = X.copy()
    for val in grid:
        X_temp[feature] = val
        preds.append(model.predict(X_temp).mean())
    return preds
~~~

**O que este código faz:**  
- congela todas as features;  
- varia apenas uma;  
- calcula a previsão média;  
- retorna a curva do PDP.

---

### 🧩 Exemplo 2 — PDP com scikit‑learn

~~~python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_valid,
    ['idade']
)
~~~

**Por que usar scikit‑learn:**  
- implementação otimizada;  
- suporte a 1D e 2D;  
- integração com pipelines.

---

### 🧩 Exemplo 3 — PDP 2D (interações)

~~~python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_valid,
    [('idade', 'renda')]
)
~~~

**Interpretação:**  
- eixo X = idade  
- eixo Y = renda  
- cores = previsão média  

---

### 🧩 Exemplo 4 — Comparando modelos

~~~python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(tree_model, X, ['idade'])
PartialDependenceDisplay.from_estimator(rf_model, X, ['idade'])
~~~

**Por que comparar:**  
- árvores → curvas em degraus  
- florestas → curvas suaves  

---

### 🧩 Exemplo 5 — PDP com dados sintéticos (exemplo adicional)

~~~python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# Dados sintéticos
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 200)

model = RandomForestRegressor(random_state=0).fit(X, y)

PartialDependenceDisplay.from_estimator(model, X, [0])
~~~

**O que este exemplo mostra:**  
- modelos não lineares capturam curvas suaves;  
- PDP revela a forma aprendida.

</details>

---

## 🟫 3.6. Integração com capítulos anteriores
<details>
<br>

| Técnica | Responde |
|--------|----------|
| **Permutation Importance** | *O que importa?* |
| **PDP** | *Como importa?* |

PDP prepara o terreno para o próximo capítulo:

> **SHAP — explicações individuais para cada previsão.**

</details>

---

## 🟪 3.7. Boas práticas e limitações
<details>
<br>

### ✔ Boas práticas
- Usar PDP apenas em regiões com dados reais.  
- Comparar modelos para entender diferenças estruturais.  
- Usar 2D PDP para investigar interações.  
- Validar se a curva faz sentido no domínio.  
- Usar grid mais denso para curvas suaves.

---

### ⚠ Limitações importantes
- Assume independência entre features.  
- Pode gerar valores irreais (ex.: latitude impossível).  
- Pode mascarar efeitos individuais (PDP = média das curvas ICE).  
- Pode ser enganado por interações fortes.  
- Não revela causalidade.  
- Pode extrapolar para regiões sem dados.  
- Pode ser instável em datasets pequenos.

</details>

---

## 🟧 3.8. Glossário técnico
<details>
<br>

- **Partial Dependence Plot (PDP)** — curva que mostra o efeito médio de uma feature.  
- **Interação** — quando o efeito de uma feature depende de outra.  
- **Grid Resolution** — número de pontos usados para construir a curva.  
- **Marginalização** — média sobre todas as outras features.  
- **Extrapolação** — prever em regiões sem dados reais.  
- **Curva em degraus** — típica de árvores de decisão.  
- **Curva suave** — típica de florestas e boosting.

</details>

---

## 🟨 3.9. Referência rápida
<details>
<br>

- PDP mostra **como** uma feature afeta a previsão.  
- Varia uma feature, mantém as outras constantes.  
- Mede efeito **marginal médio**.  
- Funciona para qualquer modelo.  
- Pode ser 1D ou 2D.  
- Não mostra efeitos individuais.  
- Assume independência entre features.  
- Complementa Permutation Importance.

</details>

---

## 🟩 3.10. Conclusão do capítulo
<details>
<br>

Partial Dependence Plots são uma das ferramentas mais importantes para interpretar modelos de machine learning.  
Eles revelam **a forma da relação** entre features e previsões, permitindo:

- validar comportamento do modelo;  
- identificar padrões não lineares;  
- investigar interações;  
- comunicar resultados de forma clara.

No próximo capítulo, avançaremos para uma técnica ainda mais poderosa:

> **SHAP — explicações individuais para cada previsão.**

<br>
</details>
</details>
<br>

---

# 📘 Capítulo 4 — SHAP Values  
<details>

### *Machine Learning Explainability — Um Guia Prático e Comentado*

<br>

---

## **4.1. Introdução**
<details>
<br>

Nos capítulos anteriores, aprendemos a extrair insights globais sobre modelos:

- **Permutation Importance** mostrou *quais* variáveis importam.  
- **Partial Dependence Plots (PDPs)** mostraram *como* cada variável afeta a previsão média.

Mas ainda falta uma peça essencial:

> **Por que o modelo fez esta previsão específica?**

Em áreas como saúde, crédito e políticas públicas, essa pergunta não é opcional — é obrigatória.  
Modelos precisam justificar decisões individuais.

É aqui que entram os **SHAP Values** (*SHapley Additive exPlanations*), uma técnica elegante e matematicamente consistente para decompor uma previsão em contribuições individuais de cada feature.

Este capítulo explica:

- a intuição por trás dos valores de Shapley;  
- como SHAP decompõe uma previsão;  
- como interpretar gráficos de força (*force plots*);  
- como SHAP complementa Permutation Importance e PDPs;  
- como preparar o leitor para o Exercise 4.

</details>

---

## **4.2. Revisão do fluxo anterior**
<details>
<br>

Até aqui, construímos uma base sólida:

### ✔ Capítulo 2 — Permutation Importance  
Aprendemos a medir **o que importa globalmente**.

### ✔ Capítulo 3 — Partial Dependence Plots  
Aprendemos **como uma feature afeta a previsão média**.

Mas PDPs não explicam:

- por que *este paciente* foi classificado como alto risco;  
- por que *este cliente* teve crédito negado;  
- por que *esta corrida* teve tarifa alta.

Para isso, precisamos de explicações **locais**, específicas para cada linha do dataset.

É exatamente o papel dos **SHAP Values**.

</details>

---

## **4.3. Apresentação do problema**
<details>
<br>

Imagine um hospital usando um modelo para prever risco de readmissão.

O modelo prevê:

> **“Paciente com 72% de chance de ser readmitido.”**

O médico pergunta:

- *“Por quê?”*  
- *“Quais fatores aumentaram esse risco?”*  
- *“Quais fatores diminuíram?”*  
- *“O que eu posso fazer a respeito?”*

Permutation Importance e PDPs não respondem isso.

Precisamos de uma técnica que:

- explique **uma previsão individual**;  
- mostre **quanto cada feature contribuiu**;  
- permita comparar pacientes;  
- seja consistente matematicamente.

Essa técnica é **SHAP**.

</details>

---

## **4.4. Conceito central — O que são SHAP Values**
<details>
<br>

SHAP Values são baseados na teoria de **valores de Shapley**, da Teoria dos Jogos Cooperativos.

A ideia é elegante:

> Cada feature é um “jogador” que contribui para a previsão.  
> O valor SHAP mede a contribuição justa de cada jogador.

### ✔ Intuição simples

Para uma previsão individual:

previsão = valor_base + soma(das contribuições de cada feature)

Onde:

- **valor_base** = previsão média do modelo  
- **contribuição de cada feature** = SHAP value  

### ✔ Exemplo conceitual

Se o modelo prevê:

- Risco previsto = 0.70  
- Valor base = 0.50  

Então a soma dos SHAP values deve ser:

0.70 – 0.50 = 0.20

Cada feature explica uma parte desse 0.20.

### ✔ O que SHAP mede

- impacto **local** de cada feature;  
- contribuição positiva (aumenta a previsão);  
- contribuição negativa (reduz a previsão);  
- decomposição exata da previsão.

### ✔ O que SHAP NÃO mede

- causalidade;  
- importância global (embora possa ser agregada);  
- efeitos médios (isso é PDP);  
- interações complexas (a menos que usemos extensões específicas).

### ✔ Por que SHAP é tão valorizado?

- é matematicamente consistente;  
- é local e global ao mesmo tempo;  
- funciona para qualquer modelo;  
- tem visualizações intuitivas;  
- é amplamente aceito em áreas reguladas (saúde, finanças).

</details>

---

## **4.5. Exemplos conceituais com código**
<details>
<br>

Os exemplos abaixo são **conceituais**, usando dados sintéticos, apenas para ilustrar o funcionamento.

---

### 🧩 Exemplo 1 — SHAP em um modelo de árvore

~~~python
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Dados sintéticos
X = np.random.randn(200, 3)
y = X[:, 0] * 3 + X[:, 1] * -2 + np.random.randn(200)

model = RandomForestRegressor(random_state=0).fit(X, y)

# Explicador SHAP para modelos de árvore
explainer = shap.TreeExplainer(model)

# SHAP values para uma única linha
row = X[5:6]
shap_values = explainer.shap_values(row)
~~~

**O que este código faz**

- Treina um modelo simples;  
- Cria um explicador SHAP específico para árvores;  
- Calcula SHAP values para uma previsão individual.

---

### 🧩 Exemplo 2 — Force Plot (explicação visual)

~~~python
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, row)
~~~

**Interpretação**

- barras em rosa: contribuições que aumentam a previsão;  
- barras em azul: contribuições que reduzem a previsão;  
- o comprimento de cada barra representa a magnitude da contribuição;  
- a soma das contribuições desloca o valor base até a previsão final.

---

### 🧩 Exemplo 3 — SHAP com KernelExplainer (modelo genérico)

~~~python
import shap

# Usando as primeiras linhas de X como background
background = X[:50]

explainer = shap.KernelExplainer(model.predict, background)
shap_values_kernel = explainer.shap_values(row)
~~~

**Quando usar**

- modelos lineares;  
- redes neurais;  
- modelos arbitrários sem estrutura de árvore.

**Observação**

- KernelExplainer é mais lento e fornece uma aproximação dos valores SHAP;  
- é útil quando não há explicador específico para o tipo de modelo.

---

### 🧩 Exemplo 4 — Decomposição da previsão

~~~python
pred = model.predict(row)[0]
base = explainer.expected_value
contrib = shap_values.sum()

print("Previsão do modelo:", pred)
print("Base + soma das contribuições:", base + contrib)
~~~

**Resultado esperado**

- As duas quantidades devem ser praticamente iguais;  
- Isso ilustra a propriedade aditiva dos valores SHAP.

</details>

---

## **4.6. Integração com capítulos anteriores**
<details>
<br>

Podemos resumir o papel de cada técnica da seguinte forma:

| Técnica                    | Pergunta principal                                  |
|---------------------------|-----------------------------------------------------|
| **Permutation Importance** | O que importa globalmente?                         |
| **PDP**                   | Como a previsão média muda ao variar uma feature?   |
| **SHAP**                  | Por que esta previsão específica aconteceu?         |

SHAP complementa as técnicas anteriores:

- **Permutation Importance** fornece uma visão global de relevância;  
- **PDP** mostra a forma média da relação entre feature e previsão;  
- **SHAP** explica, em detalhe, cada previsão individual.

No contexto do problema de readmissão hospitalar (Exercise 4), SHAP será usado para:

- explicar o risco previsto para um paciente específico;  
- destacar quais fatores aumentaram ou reduziram esse risco;  
- apoiar a comunicação com médicos e outros profissionais de saúde.

</details>

---

## **4.7. Boas práticas e limitações**
<details>
<br>

### ✔ Boas práticas

- **Usar TreeExplainer** sempre que o modelo for baseado em árvores (Random Forest, Gradient Boosting, etc.);  
- **Usar KernelExplainer** apenas quando não houver explicador específico para o tipo de modelo;  
- **Focar nas maiores contribuições** ao explicar uma previsão individual, evitando sobrecarregar o leitor com dezenas de features;  
- **Comparar pacientes ou casos** com perfis diferentes para entender padrões de risco;  
- **Combinar SHAP com outras técnicas** (Permutation Importance, PDP) para obter uma visão completa do modelo.

---

### ⚠ Limitações

- **Custo computacional**: KernelExplainer pode ser muito lento em datasets grandes;  
- **Suposições de independência**: algumas variantes de SHAP assumem independência entre features, o que pode não refletir a realidade;  
- **Interpretação em alta dimensão**: com muitas features, a leitura dos gráficos pode se tornar complexa;  
- **Não implica causalidade**: SHAP explica o comportamento do modelo, não relações causais no mundo real;  
- **Sensibilidade a dados escassos**: em regiões com poucos dados, as explicações podem ser menos confiáveis.

</details>

---

## **4.8. Glossário técnico**
<details>
<br>

- **SHAP Value** — medida da contribuição de uma feature para uma previsão individual.  
- **Base Value (valor base)** — previsão média do modelo antes de considerar as features específicas de um caso.  
- **Force Plot** — visualização que mostra como cada feature empurra a previsão para cima ou para baixo em relação ao valor base.  
- **TreeExplainer** — explicador SHAP otimizado e exato para modelos de árvore.  
- **KernelExplainer** — explicador SHAP genérico, baseado em amostragem, aplicável a qualquer modelo.  
- **Local Explanation** — explicação de uma previsão específica, para uma única linha de dados.  
- **Modelo aditivo** — modelo em que a soma das contribuições das features explica a diferença entre o valor base e a previsão.

</details>

---

## **4.9. Referência rápida**
<details>
<br>

- SHAP fornece **explicações locais** para previsões individuais;  
- A previsão é decomposta como:

  previsão ≈ valor_base + soma(dos SHAP values)

- Contribuições positivas aumentam a previsão;  
- Contribuições negativas reduzem a previsão;  
- TreeExplainer é a escolha padrão para modelos de árvore;  
- KernelExplainer é mais geral, porém mais lento;  
- SHAP complementa Permutation Importance e PDPs, conectando visão global e explicações individuais.

</details>

---

## **4.10. Conclusão do capítulo**
<details>
<br>

Neste capítulo, você viu:

- a motivação para explicações individuais de previsões;  
- a intuição por trás dos valores de Shapley;  
- como SHAP decompõe uma previsão em contribuições de cada feature;  
- exemplos conceituais de uso com modelos de árvore e explicadores genéricos;  
- como SHAP se integra às técnicas vistas anteriormente (Permutation Importance e PDPs).

No próximo passo, no **Exercise 4**, você aplicará SHAP em um cenário de readmissão hospitalar, construindo explicações individuais para pacientes e criando funções que destacam fatores de risco relevantes.

Esse exercício conecta:

- teoria;  
- prática;  
- comunicação com especialistas de domínio.

Ele marca a transição de ferramentas isoladas de explicabilidade para um fluxo completo de análise e comunicação de modelos.

<br>
</details>
</details>
<br>

---

# 📘 Capítulo 5 — SHAP Avançado  
<details>

### *Machine Learning Explainability — Um Guia Prático e Comentado*

<br>

---

## 5.1. Introdução
<details>
<br>

Nos capítulos anteriores, você aprendeu três formas complementares de extrair insights de modelos:

- **Permutation Importance** — o que importa  
- **PDPs** — como importa  
- **SHAP Values** — por que uma previsão específica foi feita  

Agora avançamos para o uso **avançado** dos valores SHAP.

A Lesson 5 do Kaggle mostra que, além de explicar previsões individuais, SHAP pode ser **agregado** para revelar:

- padrões globais  
- interações entre variáveis  
- efeitos não lineares  
- comportamentos complexos que nem PDPs nem Permutation Importance capturam

Este capítulo explica esses usos avançados, preparando você para interpretar:

- **SHAP Summary Plot**  
- **SHAP Dependence Contribution Plot**

</details>

---

## 5.2. Revisão do fluxo anterior
<details>
<br>

Até aqui, você aprendeu:

- **Permutation Importance** → o que importa  
- **PDPs** → como importa (em média)  
- **SHAP Values (básico)** → por que uma previsão específica foi feita  

Agora vamos combinar SHAP com agregações para obter:

- importância global mais detalhada  
- efeitos não lineares  
- interações entre features  
- variações individuais dentro de uma mesma feature  

Essas técnicas completam o conjunto de ferramentas de interpretabilidade.

</details>

---

## 5.3. Apresentação do problema
<details>
<br>

Mesmo com SHAP básico, ainda restam perguntas importantes:

- Uma feature tem efeito **constante** ou **variável**?  
- O efeito muda dependendo de outras features?  
- Existem **interações** que o modelo aprendeu?  
- A importância de uma feature vem de **poucos casos extremos** ou de **efeitos consistentes**?  
- Como comparar duas features com importâncias semelhantes?

Essas perguntas não são respondidas por:

- Permutation Importance  
- PDPs  
- SHAP individual  

Para isso, precisamos de **agregações de SHAP Values**.

</details>

---

## 5.4. Conceito central — SHAP Summary & Dependence Plots
<details>
<br>

### SHAP Summary Plot

É o gráfico global mais importante do SHAP. Ele mostra, em um único painel:

- importância global  
- direção do efeito  
- magnitude do efeito  
- distribuição dos efeitos  
- presença de interações  
- comportamento de valores altos e baixos  

Cada ponto representa:

- uma linha do dataset  
- uma feature  
- um SHAP value  

**Eixos:**

- eixo Y → features ordenadas por importância  
- eixo X → impacto no modelo (SHAP value)  
- cor → valor da feature (alto/baixo)

---

### SHAP Dependence Contribution Plot

É a versão avançada do PDP.

Ele mostra:

- como o valor da feature afeta o SHAP value  
- como esse efeito varia entre observações  
- como outra feature interage com ela (via cor)

Enquanto o PDP mostra **a média**, o SHAP dependence mostra **a distribuição completa**.

---

### O que essas técnicas medem

- efeitos individuais  
- efeitos globais  
- interações  
- não linearidades  
- variabilidade entre observações  

### O que elas NÃO medem

- causalidade  
- comportamento fora da distribuição de dados  
- efeitos condicionais no sentido causal

</details>

---

## 5.5. Exemplos conceituais com código
<details>
<br>

Os exemplos abaixo são **conceituais**, usando dados sintéticos, apenas para ilustrar a lógica.

---

### Exemplo 1 — SHAP Summary Plot (conceitual)

~~~python
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Dados sintéticos
X = pd.DataFrame({
    "x1": np.random.normal(0, 1, 200),
    "x2": np.random.uniform(-2, 2, 200),
})
y = 3 * X["x1"] + 0.5 * (X["x2"] ** 2) + np.random.normal(0, 0.3, 200)

model = RandomForestRegressor(random_state=0).fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
~~~

**Intuição:**

- `x1` tem efeito aproximadamente linear  
- `x2` tem efeito não linear  
- o summary plot mostra quais variáveis têm maior impacto e como seus valores empurram a previsão para cima ou para baixo  

---

### Exemplo 2 — SHAP Dependence Plot (conceitual)

~~~python
shap.dependence_plot("x2", shap_values, X, interaction_index="x1")
~~~

**Interpretação:**

- eixo X → valores de `x2`  
- eixo Y → impacto de `x2` na previsão (SHAP value)  
- cor → valores de `x1`  

Se houver interação, o padrão de cores ao longo da curva não será aleatório.

---

### Exemplo 3 — Interação forte

~~~python
X["x3"] = X["x1"] * X["x2"]
y = 3 * X["x1"] + 2 * X["x3"] + np.random.normal(0, 0.3, 200)

model = RandomForestRegressor(random_state=0).fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.dependence_plot("x1", shap_values, X, interaction_index="x2")
~~~

**O que isso mostra:**

- o efeito de `x1` depende de `x2`  
- a interação aparece como padrões de cor organizados  
- pontos com mesmo `x1` podem ter impactos diferentes dependendo de `x2`  

</details>

---

## 5.6. Integração com capítulos anteriores
<details>
<br>

| Técnica                 | Pergunta principal                                      |
|------------------------|---------------------------------------------------------|
| Permutation Importance | O que importa?                                          |
| PDP                    | Como importa (em média)?                                |
| SHAP individual        | Por que esta previsão específica foi feita?            |
| SHAP Summary           | Como cada feature afeta todas as previsões?            |
| SHAP Dependence        | Como o efeito varia entre observações? Há interações?  |

SHAP avançado combina:

- granularidade local (previsão por previsão)  
- visão global (todas as observações)  
- análise de interações e não linearidades  

Ele complementa e, em muitos casos, substitui:

- Permutation Importance como medida de importância global  
- PDPs como visualização de efeito médio  

</details>

---

## 5.7. Boas práticas e limitações
<details>
<br>

### Boas práticas

- Usar **summary plots** como primeira visão global do modelo.  
- Usar **dependence plots** para investigar features específicas e interações.  
- Focar nas features mais importantes antes de explorar as demais.  
- Usar subconjuntos de dados (amostras) quando o dataset for grande, para reduzir tempo de cálculo.  
- Sempre interpretar SHAP em conjunto com conhecimento de domínio.

---

### Limitações

- Cálculo de SHAP pode ser lento em modelos grandes ou datasets extensos.  
- Gráficos podem ser difíceis de interpretar sem prática.  
- SHAP não implica causalidade — apenas descreve o comportamento do modelo.  
- Interações sutis podem não ser óbvias visualmente.  
- Valores extremos podem dominar a escala dos gráficos.

</details>

---

## 5.8. Glossário técnico
<details>
<br>

- **SHAP Summary Plot:** gráfico que mostra, para cada feature, a distribuição dos SHAP values, sua importância global e a direção do efeito.  
- **SHAP Dependence Plot:** gráfico que mostra como o valor de uma feature se relaciona com seu SHAP value, incluindo possíveis interações via cor.  
- **Interação:** situação em que o efeito de uma feature depende do valor de outra.  
- **Efeito não linear:** relação em que a mudança na previsão não é proporcional à mudança na feature.  
- **Variabilidade de efeito:** diferença de impacto da mesma feature entre diferentes observações.  

</details>

---

## 5.9. Referência rápida
<details>
<br>

- SHAP pode ser usado tanto para explicações **locais** quanto **globais**.  
- **Summary plots** substituem, em muitos casos, gráficos de importância tradicionais.  
- **Dependence plots** são uma versão enriquecida de PDPs, com distribuição completa e interação.  
- Cores nos gráficos indicam valores de outra feature, ajudando a revelar interações.  
- SHAP avançado é ideal para entender modelos complexos em profundidade.

</details>

---

## 5.10. Conclusão do capítulo
<details>
<br>

Neste capítulo, você viu que:

- SHAP não serve apenas para explicar uma previsão individual.  
- Ao agregar SHAP values, é possível obter uma visão global rica do modelo.  
- Summary plots mostram importância, direção e distribuição dos efeitos.  
- Dependence plots revelam não linearidades e interações entre features.  

Com isso, você fecha o ciclo de técnicas de explicabilidade do curso:

- **Permutation Importance** → o que importa  
- **PDPs** → como importa em média  
- **SHAP (básico)** → por que uma previsão específica foi feita  
- **SHAP (avançado)** → como o modelo se comporta globalmente, com interações e variações individuais  

A partir daqui, o próximo passo natural é aplicar essas técnicas em problemas reais, usando datasets próprios e construindo narrativas de explicabilidade para stakeholders.

</details>
</details>
