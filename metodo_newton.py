#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho de Cálculo Numérico: Método de Newton-Raphson

Equipe de Desenvolvimento:
* Mateus de Souza Arruda
* Rayssa Conceição Santiago
* Jeanderson Athamay Araújo dos Anjos

Descrição:
    Este script implementa o Método de Newton-Raphson para encontrar a raiz
    de uma equação não linear de uma variável. O exemplo base do projeto é f(x) = x^3 - x - 2.
    O código realiza a validação de derivadas nulas para evitar divisão por zero,
    controla limites de iteração, tabula e formata os dados de convergência e gera um
    gráfico detalhado da evolução do método salvando como imagem.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def f(x):
    """
    Função matemática cuja raiz está sendo procurada.
    Exemplo base: f(x) = x^3 - x - 2
    """
    return x**3 - x - 2

def df(x):
    """
    Derivada analítica da função f(x).
    f'(x) = 3*x^2 - 1
    """
    return 3 * x**2 - 1

def metodo_newton_raphson(x0, Es, N0):
    """
    Executa o Método de Newton-Raphson para aproximação de raízes.

    Parâmetros:
        x0 (float): Estimativa inicial (chute inicial).
        Es (float): Tolerância do erro relativo aproximado percentual (%).
        N0 (int): Número máximo de iterações permitidas.

    Retorna:
        iteracoes (list): Histórico de iterações, onde cada elemento é um dicionário.
        sucesso (bool): True se a tolerância de erro foi atingida, False caso contrário.
        raiz (float): A última estimativa calculada da raiz.
        mensagem (str): Descrição textual do resultado final da execução.
    """
    iteracoes = []
    x_atual = x0
    sucesso = False
    mensagem = ""

    for i in range(1, N0 + 1):
        fx = f(x_atual)
        dfx = df(x_atual)

        # Validação contra divisão por zero (derivada nula ou muito próxima de zero)
        if abs(dfx) < 1e-12:
            mensagem = f"Erro de Instabilidade: Derivada nula ou muito próxima de zero (f'(x) = {dfx:.4e}) na iteração {i}. Execução abortada para evitar divisão por zero."
            break

        x_novo = x_atual - fx / dfx             # Fórmula Principal do Método de Newton-Raphson

        # Cálculo do erro relativo percentual aproximado (E_a)
        # Evita divisão por zero/falsa convergência usando max(abs(x_novo), 1e-15) no denominador
        ea = abs((x_novo - x_atual) / max(abs(x_novo), 1e-15)) * 100

        iteracoes.append({
            'Iteracao': i,
            'xi': x_atual,
            'f_xi': fx,
            'df_xi': dfx,
            'xi_mais_1': x_novo,
            'Erro_Percentual': ea
        })

        # Critério de parada: erro menor que a tolerância estipulada
        if ea < Es:
            sucesso = True
            mensagem = f"Convergência alcançada com sucesso! Erro aproximado {ea:.6f}% < tolerância {Es}%."
            x_atual = x_novo
            break

        x_atual = x_novo
    else:
        mensagem = f"Aviso de Limite: O número máximo de {N0} iterações foi atingido sem alcançar a convergência da tolerância desejada."

    return iteracoes, sucesso, x_atual, mensagem

def gerar_grafico(iteracoes, raiz):
    """
    Gera o gráfico contendo a curva da função, a raiz encontrada e as retas tangentes das iterações.
    Salva o arquivo final como 'grafico_newton.png'.
    """
    # Coleta todos os pontos X de interesse para ajustar os limites do plano cartesiano
    passos_plot = min(len(iteracoes), 5)
    pontos_x = [raiz]
    for idx in range(passos_plot):
        pontos_x.append(iteracoes[idx]['xi'])
        pontos_x.append(iteracoes[idx]['xi_mais_1'])
    
    x_min_pontos = min(pontos_x)
    x_max_pontos = max(pontos_x)
    
    # Define limites horizontais com margem proporcional dinâmica (mínimo de 1.5 de folga nas laterais)
    largura = x_max_pontos - x_min_pontos
    folga = max(largura * 0.15, 1.5)
    x_min = x_min_pontos - folga
    x_max = x_max_pontos + folga

    # Define o intervalo do domínio para plotar a curva principal com base nos limites dinâmicos
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = f(x_vals)

    # Configuração estética do gráfico
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - x - 2$', color='#1f77b4', linewidth=2.5)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
    
    # Destaca a raiz aproximada encontrada
    plt.scatter(raiz, 0, color='#d62728', marker='*', s=200, zorder=5, 
                label=f'Raiz Aproximada: {raiz:.8f}')

    # Desenha os passos de iteração (projeções verticais e tangentes)
    # Mostra no máximo as 5 primeiras iterações para evitar poluição visual do gráfico
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, passos_plot))

    for idx in range(passos_plot):
        it = iteracoes[idx]
        xi = it['xi']
        f_xi = it['f_xi']
        df_xi = it['df_xi']
        xi_next = it['xi_mais_1']
        color = colors[idx]

        # Linha vertical tracejada do eixo X até a curva: (xi, 0) -> (xi, f(xi))
        plt.plot([xi, xi], [0, f_xi], color=color, linestyle='--', linewidth=1.2, alpha=0.7)
        plt.scatter(xi, f_xi, color=color, s=50, zorder=4)

        # Reta tangente: ligando (xi, f(xi)) ao ponto de interseção no eixo X (xi_next, 0)
        # Reta definida por: y = f(xi) + f'(xi)*(x - xi)
        t_vals = np.array([xi, xi_next])
        yt_vals = f_xi + df_xi * (t_vals - xi)
        plt.plot(t_vals, yt_vals, color=color, linestyle='-', linewidth=1.5, alpha=0.85,
                 label=f'Iteração {it["Iteracao"]} (x = {xi:.3f})')

    # Limita o eixo Y proporcionalmente para focar na região onde as iterações ocorrem
    pontos_y = [0.0]
    for idx in range(passos_plot):
        pontos_y.append(iteracoes[idx]['f_xi'])
    
    y_min_pontos = min(pontos_y)
    y_max_pontos = max(pontos_y)
    
    altura = y_max_pontos - y_min_pontos
    folga_y = max(altura * 0.15, 2.0)
    y_min = y_min_pontos - folga_y
    y_max = y_max_pontos + folga_y

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title('Método de Newton-Raphson: Convergência da Raiz', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('x (Domínio)', fontsize=12)
    plt.ylabel('y (f(x))', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()

    # Salva o arquivo no diretório raiz do repositório
    plt.savefig('grafico_newton.png', dpi=300)
    plt.close()

def salvar_log(x0, Es, N0, iteracoes, sucesso, raiz, mensagem):
    """
    Grava os resultados detalhados e o histórico de iterações em um arquivo de texto.
    """
    cabecalho = (
        "================================================================================\n"
        "TRABALHO DE CÁLCULO NUMÉRICO: MÉTODO DE NEWTON-RAPHSON\n"
        "================================================================================\n"
        "Equipe de Desenvolvimento:\n"
        "* Mateus de Souza Arruda\n"
        "* Rayssa Conceição Santiago\n"
        "* Jeanderson Athamay Araújo dos Anjos\n"
        "--------------------------------------------------------------------------------\n"
    )

    parametros = (
        f"PARÂMETROS DE EXECUÇÃO:\n"
        f"- Estimativa Inicial (x0): {x0}\n"
        f"- Tolerância de Erro (Es): {Es}%\n"
        f"- Limite Máximo de Iterações (N0): {N0}\n"
        f"--------------------------------------------------------------------------------\n"
    )

    # Estruturação da tabela de dados para salvar no arquivo
    table_data = []
    for it in iteracoes:
        table_data.append([
            it['Iteracao'],
            f"{it['xi']:.8f}",
            f"{it['f_xi']:.8f}",
            f"{it['df_xi']:.8f}",
            f"{it['xi_mais_1']:.8f}",
            f"{it['Erro_Percentual']:.6f}%"
        ])

    headers = ["Iteração", "x_i", "f(x_i)", "f'(x_i)", "x_{i+1}", "Erro Rel. (%)"]
    tabela_str = tabulate(table_data, headers=headers, tablefmt="grid")

    resumo = (
        f"\n--------------------------------------------------------------------------------\n"
        f"RESUMO DO RESULTADO FINAL:\n"
        f"- Status de Execução: {'SUCESSO' if sucesso else 'FALHA/AVISO'}\n"
        f"- Mensagem: {mensagem}\n"
        f"- Raiz Aproximada Final: {raiz:.8f}\n"
        f"- Valor de f(raiz): {f(raiz):.4e}\n"
        f"- Total de Iterações Executadas: {len(iteracoes)}\n"
        f"================================================================================\n"
    )

    with open('resultados_newton.txt', 'w', encoding='utf-8') as f_out:
        f_out.write(cabecalho)
        f_out.write(parametros)
        f_out.write(tabela_str)
        f_out.write(resumo)

def main():
    print("================================================================================")
    print("         MÉTODO DE NEWTON-RAPHSON: BUSCA DE RAÍZES NÃO LINEARES")
    print("================================================================================")
    print("Função de Trabalho: f(x) = x^3 - x - 2 = 0")
    print("--------------------------------------------------------------------------------")

    # Coleta e validação interativa dos parâmetros do usuário
    try:
        x0 = float(input("Digite a aproximação inicial (x0): "))
    except ValueError:
        print("Erro: A aproximação inicial deve ser um valor numérico real.")
        return

    try:
        Es = float(input("Digite a tolerância de erro tolerada (Es em %): "))
        if Es <= 0:
            print("Erro: A tolerância deve ser um valor numérico estritamente positivo.")
            return
    except ValueError:
        print("Erro: A tolerância de erro deve ser um valor numérico real.")
        return

    try:
        N0 = int(input("Digite o limite máximo de iterações de segurança (N0): "))
        if N0 <= 0:
            print("Erro: O limite de iterações deve ser um número inteiro estritamente positivo.")
            return
    except ValueError:
        print("Erro: O número de iterações deve ser um valor inteiro.")
        return

    print("\nProcessando iterações do Método de Newton-Raphson...")
    iteracoes, sucesso, raiz, mensagem = metodo_newton_raphson(x0, Es, N0)

    # Exibição dos resultados em tabela formatada na tela
    print("\nTabela de Convergência Iterativa:")
    table_print = []
    for it in iteracoes:
        table_print.append([
            it['Iteracao'],
            f"{it['xi']:.6f}",
            f"{it['f_xi']:.6f}",
            f"{it['df_xi']:.6f}",
            f"{it['xi_mais_1']:.6f}",
            f"{it['Erro_Percentual']:.6f}%"
        ])
    
    headers = ["Iteração", "x_i", "f(x_i)", "f'(x_i)", "x_{i+1}", "Erro Rel. (%)"]
    print(tabulate(table_print, headers=headers, tablefmt="fancy_grid"))

    print(f"\nResultado da Execução: {mensagem}")
    print(f"Raiz aproximada obtida: {raiz:.8f}")

    # Geração dos arquivos de saída (log e gráfico)
    try:
        salvar_log(x0, Es, N0, iteracoes, sucesso, raiz, mensagem)
        print("-> Tabela completa e resumo salvos em 'resultados_newton.txt'.")
    except Exception as err:
        print(f"Erro ao salvar arquivo de texto 'resultados_newton.txt': {err}")

    try:
        gerar_grafico(iteracoes, raiz)
        print("-> Gráfico ilustrativo de convergência salvo em 'grafico_newton.png'.")
    except Exception as err:
        print(f"Erro ao gerar/salvar imagem 'grafico_newton.png': {err}")

    print("================================================================================")

if __name__ == "__main__":
    main()
