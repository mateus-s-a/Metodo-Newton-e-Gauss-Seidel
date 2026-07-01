# Trabalho de Cálculo Numérico: Métodos de Newton-Raphson e Gauss-Seidel

**Equipe de Desenvolvimento:**
* Mateus de Souza Arruda
* Rayssa Conceição Santiago
* Jeanderson Athamay Araújo dos Anjos

---

## Descrição do Projeto
Este repositório contém as implementações computacionais em Python de dois métodos numéricos clássicos da disciplina de Cálculo Numérico:
1. **Método de Newton-Raphson:** Um método iterativo para encontrar raízes de equações não lineares.
2. **Método de Gauss-Seidel:** Um método iterativo para a resolução de sistemas de equações lineares.

A proposta deste projeto é desenvolver ambas as soluções de forma independente (dois arquivos Python separados), estruturando códigos robustos com validação matemática rigorosa contra divisão por zero, loops infinitos por divergência e critérios estritos de convergência.

---

## 1. Método de Newton-Raphson (Busca de Raízes)

O método de Newton-Raphson é um dos algoritmos de busca de raízes mais eficientes e amplamente utilizados devido à sua convergência quadrática. Ele se baseia na expansão em série de Taylor de primeira ordem da função $f(x)$ em torno de uma estimativa inicial $x_0$.

### Modelo Matemático e Iterativo
Graficamente, o método substitui a curva da função pela sua reta tangente no ponto atual $x_i$. A interseção dessa reta tangente com o eixo horizontal $x$ fornece a próxima aproximação $x_{i+1}$.

A equação iterativa é definida por:

$$x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}$$

*Onde:*
*   $x_i$: estimativa atual da raiz.
*   $x_{i+1}$: próxima estimativa refinada da raiz.
*   $f(x_i)$: valor da função em $x_i$.
*   $f'(x_i)$: derivada primeira da função avaliada em $x_i$ (representa o coeficiente angular da reta tangente).

### Critério de Parada
O critério de convergência baseia-se no **Erro Relativo Percentual Aproximado ($\mathcal{E}_a$)**:

$$\mathcal{E}_a = \left| \frac{x_{i+1} - x_i}{x_{i+1}} \right| \times 100\%$$

O algoritmo encerra quando $\mathcal{E}_a < \mathcal{E}_s$ (onde $\mathcal{E}_s$ é a tolerância percentual estipulada pelo usuário) ou quando o número máximo de iterações $N_0$ é atingido.

### Exemplo Base do Projeto
Para demonstrar o funcionamento do script, utilizaremos a clássica função polinomial não linear:

$$f(x) = x^3 - x - 2 = 0$$

Cuja derivada analítica é:

$$f'(x) = 3x^2 - 1$$

Substituindo na fórmula geral de Newton-Raphson, a iteração simplificada se torna:

$$x_{i+1} = x_i - \frac{x_i^3 - x_i - 2}{3x_i^2 - 1}$$

### Exemplos Interessantes para Demonstração

Para fins de apresentação e análise de comportamento matemático de Newton-Raphson com a função de trabalho $f(x) = x^3 - x - 2$, recomenda-se testar os seguintes cenários:

#### 1. Caso Ideal: Convergência Quadrática Rápida
*   **Parâmetros de Entrada:**
    *   Chute Inicial ($x_0$): `2.0` (ou `1.5`)
    *   Tolerância ($E_s$): `1e-6` ($0.000001\%$)
    *   Iterações Máximas ($N_0$): `10`
*   **Comportamento:** O método converge com sucesso absoluto em apenas **5 iterações**. Isso ocorre porque o chute inicial está próximo à raiz real ($\approx 1.52138$) e em uma região estável e convexa da função.
*   **Significado Matemático:** Ilustra a principal vantagem prática do Método de Newton-Raphson: quando bem posicionado, o número de algarismos significativos corretos dobra a cada iteração.

#### 2. Caso de Instabilidade: Fenômeno do "Salto" (Overshoot)
*   **Parâmetros de Entrada:**
    *   Chute Inicial ($x_0$): `0.57` *(ponto muito próximo ao extremo local mínimo em $x \approx 0.57735$, onde a derivada $f'(x)$ tende a zero)*
    *   Tolerância ($E_s$): `1e-6` ($0.000001\%$)
    *   Iterações Máximas ($N_0$): `50`
*   **Comportamento:** Por conta de a derivada ser extremamente próxima de zero ($f'(0.57) \approx -0.0253$), a reta tangente é quase horizontal, projetando a aproximação seguinte para muito longe: **$x_1 \approx -93.69$**. A partir desse valor distante, o algoritmo leva **24 iterações** para retornar e convergir para a raiz exata.
*   **Significado Matemático:** Demonstra graficamente a sensibilidade do método ao chute inicial. Zonas de derivadas nulas ou quase nulas fazem a reta tangente "chutar" as iterações para fora da área de interesse, retardando consideravelmente a convergência ou provocando divergência se o limite $N_0$ for muito restrito.

### Visualização Gráfica
Ao término da execução, o script gera e salva automaticamente o gráfico `grafico_newton.png`. Este gráfico plota a curva da função $f(x) = x^3 - x - 2$, destaca no eixo $x$ o ponto exato da raiz encontrada pelo método de Newton-Raphson, e ilustra os passos ou retas tangentes de convergência.

---

## 2. Método de Gauss-Seidel (Sistemas Lineares)

O método de Gauss-Seidel é um método iterativo utilizado para resolver sistemas de equações lineares do tipo $A x = b$. Diferente do método de Jacobi, o método de Gauss-Seidel utiliza os valores recém-calculados das variáveis assim que eles se tornam disponíveis na iteração corrente, acelerando significativamente a convergência.

### Modelo Matemático e Iterativo
Para um sistema linear de $n$ equações com $n$ incógnitas, isolamos cada variável $x_i$ da diagonal principal na $i$-ésima equação. A fórmula iterativa para a $k$-ésima iteração é:

$$x_i^{(k+1)} = \frac{b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^n a_{ij} x_j^{(k)}}{a_{ii}}$$

Note que os elementos já atualizados na iteração atual $x_1^{(k+1)}, \dots, x_{i-1}^{(k+1)}$ são inseridos imediatamente no somatório da esquerda.

### Garantia de Convergência (Critério da Diagonal Dominante)
Uma condição suficiente (mas não necessária) para a convergência de Gauss-Seidel é que a matriz dos coeficientes $A$ seja **estritamente diagonal dominante**, o que significa que o valor absoluto do termo da diagonal principal de cada linha é maior que a soma dos valores absolutos de todos os outros termos da mesma linha:

$$|a_{ii}| > \sum_{j \neq i} |a_{ij}|, \quad \forall i = 1, 2, \dots, n$$

### Critério de Parada
O critério de parada monitora a variação de todas as incógnitas simultaneamente. O erro relativo percentual aproximado é calculado para cada variável $i$:

$$\mathcal{E}_{a, i} = \left| \frac{x_i^{(k+1)} - x_i^{(k)}}{x_i^{(k+1)}} \right| \times 100\%$$

O processo iterativo é encerrado com sucesso quando o maior erro relativo entre todas as variáveis for inferior à tolerância $\mathcal{E}_s$ fornecida:

$$\max \left( \mathcal{E}_{a, 1}, \mathcal{E}_{a, 2}, \dots, \mathcal{E}_{a, n} \right) < \mathcal{E}_s$$

### Exemplo Base do Projeto
Utilizaremos um sistema linear $3 \times 3$ com dominância diagonal garantida:

$$\begin{cases}
10x_1 + 2x_2 - x_3 = 27 \\
-3x_1 - 6x_2 + 2x_3 = -61.5 \\
x_1 + x_2 + 5x_3 = -21.5
\end{cases}$$

Escrevendo as equações de forma iterativa isolando as incógnitas da diagonal:

$$x_1^{(k+1)} = \frac{27 - 2x_2^{(k)} + x_3^{(k)}}{10}$$

$$x_2^{(k+1)} = \frac{-61.5 + 3x_1^{(k+1)} - 2x_3^{(k)}}{-6}$$

$$x_3^{(k+1)} = \frac{-21.5 - x_1^{(k+1)} - x_2^{(k+1)}}{5}$$

### Exemplos Interessantes para Demonstração

Para fins de demonstração do Método de Gauss-Seidel, recomenda-se testar os seguintes cenários:

#### 1. Caso de Independência do Chute Inicial (Sistema Padrão)
*   **Parâmetros de Entrada:**
    *   Escolha do Sistema: Padrão (`P`)
    *   Chute Inicial ($x_0$): `[1000.0, -1000.0, 1000.0]` *(chute inicial muito distante da solução real)*
    *   Tolerância ($E_s$): `1e-5` ($0.00001\%$)
    *   Iterações Máximas ($N_0$): `100`
*   **Comportamento:** O algoritmo converge perfeitamente para a solução exata $x = [3.0, 12.5, -7.0]$ em apenas **12 iterações**.
*   **Significado Matemático:** Comprova graficamente a robustez e a propriedade de convergência global do método. Quando a matriz é estritamente diagonal dominante, o algoritmo garante a convergência independente de quão absurdo ou distante seja o chute inicial escolhido.

#### 2. Caso de Matriz não Diagonal Dominante, mas Convergente (Sassenfeld)
*   **Parâmetros de Entrada:**
    *   Escolha do Sistema: Customizado (`C`)
    *   Ordem do Sistema ($n$): `3`
    *   Linhas da Matriz $A$:
        *   Linha 1: `2 1 0`
        *   Linha 2: `3 4 2`
        *   Linha 3: `0 1 2`
    *   Vetor $b$: `5 17 5` *(Solução exata: $[1.0, 3.0, 1.0]$)*
    *   Equações Iterativas:
        $$
        \begin{cases}
        2x_1 + x_2 = 5 \\
        3x_1 + 4x_2 + 2x_3 = 17 \\
        x_2 + 2x_3 = 5
        \end{cases}
        $$
        $$
        x_1^{(k+1)} = \frac{5 - x_2^{(k)}}{2}\\
        x_2^{(k+1)} = \frac{17 - 3x_1^{(k+1)} - 2x_3^{(k)}}{4}\\
        x_3^{(k+1)} = \frac{5 - x_2^{(k+1)}}{2}
        $$
    *   Chute Inicial ($x_0$): `[0.0, 0.0, 0.0]`
    *   Tolerância ($E_s$): `1e-4`
    *   Iterações Máximas ($N_0$): `100`
*   **Comportamento:** O script detectará e alertará que a matriz **não** atende ao critério clássico de dominância diagonal (na segunda linha, $|4| \le |3| + |2|$). No entanto, o script também calculará e mostrará que o **Critério de Sassenfeld** é atendido ($\beta_{max} = 0.875 < 1.0$), prosseguindo para a execução e convergência com sucesso absoluto em **10 iterações**.
*   **Significado Matemático:** Demonstra a utilidade do Critério de Sassenfeld como uma ferramenta mais geral e robusta do que a dominância diagonal por linhas clássica, permitindo resolver numericamente sistemas que seriam descartados por análises mais simples.

#### 3. Caso de Reordenação Automática de Linhas (Zeros na Diagonal)
*   **Parâmetros de Entrada:**
    *   Escolha do Sistema: Customizado (`C`)
    *   Ordem do Sistema ($n$): `2`
    *   Linhas da Matriz $A$:
        *   Linha 1: `0 2`
        *   Linha 2: `3 1`
    *   Vetor $b$: `4 5` *(Solução exata: $[1.0, 2.0]$)*
    *   Chute Inicial ($x_0$): `[0.0, 0.0]`
    *   Tolerância ($E_s$): `1e-5`
    *   Iterações Máximas ($N_0$): `50`
*   **Comportamento:** O script detectará imediatamente a presença de um zero na diagonal principal ($a_{11} = 0$), o que impediria a divisão na iteração clássica de Gauss-Seidel, pois sofreria uma divisão por zero na 1ª iteração do método. O script executará a reordenação automática de linhas (permutando a linha 1 com a linha 2), resultando em um sistema equivalente com diagonal estritamente dominante e livre de zeros:
    $$A' = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}, \quad b' = \begin{pmatrix} 5 \\ 4 \end{pmatrix}$$
    A partir disso, o método converge com sucesso absoluto em apenas **3 iterações**.
*   **Significado Matemático:** Demonstra a robustez computacional necessária na engenharia de algoritmos numéricos. O tratamento dinâmico de pivôs evita exceções de divisão por zero (indefinições matemáticas) e recupera sistemas teoricamente convergentes mesmo que fornecidos em ordem desfavorável.

### Visualização Gráfica
Ao término da execução, o script gera e salva automaticamente o gráfico `grafico_gauss_seidel.png`. Este gráfico ilustra a evolução da estimativa de cada variável ($x_1, x_2, x_3$) ou do erro relativo aproximado a cada iteração, fornecendo uma visão clara do comportamento e velocidade de convergência do método.

---

## Estrutura do Repositório
*   [`metodo_newton.py`](metodo_newton.py): Script contendo a lógica iterativa de busca de raízes, tratamento de erro de derivada nula, geração de gráfico e exportação de logs.
*   [`metodo_gauss_seidel.py`](metodo_gauss_seidel.py): Script contendo a implementação do sistema linear, verificação de diagonal dominante, geração de gráfico e exportação de logs.
*   `resultados_newton.txt`: Log gerado automaticamente com a tabela de convergência do método de Newton.
*   `resultados_gauss_seidel.txt`: Log gerado automaticamente com a tabela de iterações do método de Gauss-Seidel.
*   `grafico_newton.png`: Gráfico gerado contendo o comportamento da função e a convergência da raiz.
*   `grafico_gauss_seidel.png`: Gráfico gerado contendo o histórico de convergência das variáveis ou do erro.
*   [`requirements.txt`](requirements.txt): Arquivo com as dependências externas do projeto (`matplotlib`, `numpy`, `tabulate`).

---

## Como Executar

### Pré-requisitos
1. Certifique-se de ter o Python 3 instalado.
2. Ative o ambiente virtual contido na pasta raiz do repositório:
   ```bash
   source .venv/bin/activate
   ```
3. Instale as dependências listadas no arquivo `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Executando o Método de Newton-Raphson
Execute o script correspondente:
```bash
python metodo_newton.py
```
O console solicitará:
*   A aproximação inicial ($x_0$).
*   A tolerância de erro tolerada ($\mathcal{E}_s$ em %).
*   O número máximo de iterações de segurança ($N_0$).

### Executando o Método de Gauss-Seidel
Execute o script correspondente:
```bash
python metodo_gauss_seidel.py
```
O console solicitará:
*   Os chutes iniciais para cada variável ($x_1, x_2, x_3$).
*   A tolerância de erro tolerada ($\mathcal{E}_s$ em %).
*   O limite máximo de iterações ($N_0$).