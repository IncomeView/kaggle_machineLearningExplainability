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

# 3