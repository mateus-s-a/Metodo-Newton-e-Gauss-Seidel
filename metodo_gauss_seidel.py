#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho de Cálculo Numérico: Método de Gauss-Seidel

Equipe de Desenvolvimento:
* Mateus de Souza Arruda
* Rayssa Conceição Santiago
* Jeanderson Athamay Araújo dos Anjos

Descrição:
    Este script implementa o Método de Gauss-Seidel para resolver sistemas
    de equações lineares de dimensões arbitrárias. O exemplo padrão de execução
    é o sistema 3x3 diagonal dominante definido no README.
    O código verifica critérios de dominância diagonal e Sassenfeld, avisa se a 
    convergência não for garantida com opção de continuar, realiza pivoteamento de
    equações em caso de diagonais nulas, formata logs em tabelas de texto e console,
    e plota um gráfico duplo contendo a evolução das variáveis e o decaimento do erro.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def check_diagonal_dominance(A):
    """
    Verifica se a matriz A é estritamente diagonal dominante.
    Retorna True se for, False caso contrário.
    """
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        soma_linha = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= soma_linha:
            return False
    return True

def check_sassenfeld(A):
    """
    Verifica se a matriz A atende ao critério de convergência de Sassenfeld.
    Retorna (converte, max_beta)
    """
    n = len(A)
    betas = []
    for i in range(n):
        soma_1 = sum(abs(A[i][j]) * betas[j] for j in range(i))
        soma_2 = sum(abs(A[i][j]) for j in range(i + 1, n))
        beta_i = (soma_1 + soma_2) / abs(A[i][i])
        betas.append(beta_i)
    max_beta = max(betas)
    return max_beta < 1.0, max_beta

def reordenar_sistema(A, b):
    """
    Tenta reordenar as equações (linhas) para que nenhum elemento da diagonal principal seja zero.
    Retorna (A_nova, b_novo) ou lança ValueError se impossível.
    """
    import itertools
    n = len(A)
    indices_linhas = list(range(n))
    for perm in itertools.permutations(indices_linhas):
        diag_ok = all(abs(A[p][i]) > 1e-12 for i, p in enumerate(perm))
        if diag_ok:
            A_nova = [A[p] for p in perm]
            b_novo = [b[p] for p in perm]
            return A_nova, b_novo
    raise ValueError("Erro Crítico: Não foi possível reordenar o sistema para remover zeros da diagonal principal.")

def metodo_gauss_seidel(A, b, x0, Es, N0):
    """
    Executa o Método de Gauss-Seidel para resolver o sistema linear Ax = b.

    Parâmetros:
        A (list of lists): Matriz de coeficientes.
        b (list): Vetor de termos independentes.
        x0 (list): Chute inicial para cada incógnita.
        Es (float): Tolerância do erro relativo percentual aproximado máximo (%).
        N0 (int): Limite máximo de iterações permitidas.

    Retorna:
        iteracoes (list): Histórico detalhado das iterações executadas.
        sucesso (bool): True se convergiu, False caso contrário.
        x_final (list): Solução final aproximada.
        mensagem (str): Descrição textual do encerramento.
    """
    n = len(A)
    # Validação de dimensões
    if len(b) != n or len(x0) != n or any(len(row) != n for row in A):
        raise ValueError("Erro de Dimensões: As dimensões de A, b e x0 devem ser compatíveis e A deve ser uma matriz quadrada.")

    x = list(x0)
    iteracoes = []
    sucesso = False
    mensagem = ""

    # Verifica se há elementos nulos na diagonal e tenta reordenar
    if any(abs(A[i][i]) < 1e-12 for i in range(n)):
        try:
            A, b = reordenar_sistema(A, b)
        except ValueError as err:
            return [], False, x0, str(err)

    for k in range(1, N0 + 1):
        x_velho = list(x)
        erros_etapa = []

        for i in range(n):
            # Gauss-Seidel utiliza os valores mais recentes de x_j (atualizados na iteração atual)
            soma = 0.0
            for j in range(n):
                if j != i:
                    soma += A[i][j] * x[j]

            x[i] = (b[i] - soma) / A[i][i]

            # Erro relativo percentual aproximado para a variável x_i
            # Evita divisão por zero/falsa convergência usando max(abs(x[i]), 1e-15) no denominador
            ea_i = abs((x[i] - x_velho[i]) / max(abs(x[i]), 1e-15)) * 100
            erros_etapa.append(ea_i)

        erro_max = max(erros_etapa)

        iteracoes.append({
            'Iteracao': k,
            'x': list(x),
            'erros': list(erros_etapa),
            'Erro_Maximo': erro_max
        })

        if erro_max < Es:
            sucesso = True
            mensagem = f"Convergência alcançada com sucesso! Maior erro {erro_max:.6f}% < tolerância {Es}%."
            break
    else:
        mensagem = f"Aviso de Limite: Limite de {N0} iterações foi atingido sem satisfazer a tolerância estipulada."

    return iteracoes, sucesso, x, mensagem

def gerar_graficos(iteracoes, n, x0=None):
    """
    Gera uma imagem de alta qualidade contendo dois subplots de convergência.
    Salva como 'grafico_gauss_seidel.png'.
    """
    iters = [it['Iteracao'] for it in iteracoes]
    erro_max = [it['Erro_Maximo'] for it in iteracoes]
    
    # Extrai o histórico dos valores de cada incógnita x_i incluindo o chute inicial (iteração 0)
    if x0 is not None:
        iters_x = [0] + iters
        x_historico = {i: [x0[i]] + [it['x'][i] for it in iteracoes] for i in range(n)}
    else:
        iters_x = iters
        x_historico = {i: [it['x'][i] for it in iteracoes] for i in range(n)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot 1: Trajetória dos valores das variáveis
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(n):
        ax1.plot(iters_x, x_historico[i], marker='o', markersize=4,
                 label=f'$x_{i+1}$', color=colors[i % len(colors)], linewidth=2)
    ax1.set_title('Evolução das Estimativas das Incógnitas ($x_i$)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iterações', fontsize=10)
    ax1.set_ylabel('Valor Estimado', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best', frameon=True, shadow=True)

    # Subplot 2: Decaimento do erro máximo
    ax2.plot(iters, erro_max, marker='s', markersize=4, color='#d62728', 
             linewidth=2, label=r'$\max(\mathcal{E}_a)$')
    ax2.set_title('Decaimento do Erro Relativo Máximo', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iterações', fontsize=10)
    ax2.set_ylabel('Erro Aproximado (%)', fontsize=10)

    # Aplica escala logarítmica para o erro se houver variação suficiente
    if len(iters) > 1 and max(erro_max) > 10 * min(erro_max) and min(erro_max) > 0:
        ax2.set_yscale('log')
        ax2.set_ylabel('Erro Aproximado (%) - Escala Log', fontsize=10)
        
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.legend(loc='best', frameon=True, shadow=True)

    plt.suptitle('Método de Gauss-Seidel: Análise de Convergência', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Salva a imagem na raiz do repositório
    plt.savefig('grafico_gauss_seidel.png', dpi=300)
    plt.close()

def salvar_log(A, b, x0, Es, N0, iteracoes, sucesso, x_final, mensagem):
    """
    Exporta a tabela de iterações e resumo detalhado em 'resultados_gauss_seidel.txt'.
    """
    n = len(A)
    cabecalho = (
        "================================================================================\n"
        "TRABALHO DE CÁLCULO NUMÉRICO: MÉTODO DE GAUSS-SEIDEL\n"
        "================================================================================\n"
        "Equipe de Desenvolvimento:\n"
        "* Mateus de Souza Arruda\n"
        "* Rayssa Conceição Santiago\n"
        "* Jeanderson Athamay Araújo dos Anjos\n"
        "--------------------------------------------------------------------------------\n"
    )

    # Transcreve o sistema de equações no arquivo de log
    sist_str = "SISTEMA LINEAR SOLUCIONADO:\n"
    for i in range(n):
        termos = []
        for j in range(n):
            coef = A[i][j]
            sinal = "+" if coef >= 0 else "-"
            valor = abs(coef)
            if j == 0:
                termos.append(f"{coef}*x1")
            else:
                termos.append(f"{sinal} {valor}*x{j+1}")
        sist_str += f"  {' '.join(termos)} = {b[i]}\n"
    sist_str += "--------------------------------------------------------------------------------\n"

    parametros = (
        f"PARÂMETROS DE EXECUÇÃO:\n"
        f"- Chutes Iniciais (x0): {x0}\n"
        f"- Tolerância de Erro (Es): {Es}%\n"
        f"- Limite Máximo de Iterações (N0): {N0}\n"
        f"--------------------------------------------------------------------------------\n"
    )

    # Constrói a tabela de iterações detalhada
    table_data = []
    for it in iteracoes:
        linha = [it['Iteracao']]
        # Adiciona valores estimados das incógnitas nesta iteração
        for val in it['x']:
            linha.append(f"{val:.8f}")
        # Adiciona erros relativos das incógnitas
        for err in it['erros']:
            linha.append(f"{err:.6f}%")
        # Adiciona o maior erro da iteração
        linha.append(f"{it['Erro_Maximo']:.6f}%")
        table_data.append(linha)

    # Cabeçalho dinâmico das colunas
    headers = ["Iter."]
    for i in range(n):
        headers.append(f"x{i+1}")
    for i in range(n):
        headers.append(f"Erro x{i+1}")
    headers.append("Erro Máx")

    tabela_str = tabulate(table_data, headers=headers, tablefmt="grid")

    sol_str = ", ".join(f"x{i+1} = {x_final[i]:.8f}" for i in range(n))
    resumo = (
        f"\n--------------------------------------------------------------------------------\n"
        f"RESUMO DO RESULTADO FINAL:\n"
        f"- Status de Execução: {'SUCESSO' if sucesso else 'FALHA/AVISO'}\n"
        f"- Mensagem: {mensagem}\n"
        f"- Solução Vetorial Encontrada: [{sol_str}]\n"
        f"- Total de Iterações Executadas: {len(iteracoes)}\n"
        f"================================================================================\n"
    )

    with open('resultados_gauss_seidel.txt', 'w', encoding='utf-8') as f_out:
        f_out.write(cabecalho)
        f_out.write(sist_str)
        f_out.write(parametros)
        f_out.write(tabela_str)
        f_out.write(resumo)

def main():
    print("================================================================================")
    print("            MÉTODO DE GAUSS-SEIDEL: RESOLUÇÃO DE SISTEMAS LINEARES")
    print("================================================================================")
    
    # Sistema linear padrão sugerido no README
    A_padrao = [
        [10.0, 2.0, -1.0],
        [-3.0, -6.0, 2.0],
        [1.0, 1.0, 5.0]
    ]
    b_padrao = [27.0, -61.5, -21.5]

    print("Sistema Padrão:")
    print("  (1)  10*x1 + 2*x2  -   x3 = 27.0")
    print("  (2)  -3*x1 - 6*x2  + 2*x3 = -61.5")
    print("  (3)     x1 +   x2  + 5*x3 = -21.5")
    print("--------------------------------------------------------------------------------")

    escolha = input("Deseja resolver o sistema padrão (P) ou inserir um sistema customizado (C)? [P/C]: ").strip().upper()
    
    if escolha == 'C':
        try:
            n = int(input("Digite o número de variáveis/equações (n): "))
            if n <= 1:
                print("Erro: A ordem do sistema deve ser maior que 1.")
                return
            
            A = []
            b = []
            print("\nDigite os coeficientes da matriz A linha por linha (separados por espaço):")
            for i in range(n):
                linha = list(map(float, input(f"Linha {i+1} (deve conter {n} números): ").split()))
                if len(linha) != n:
                    print(f"Erro: A linha {i+1} deve conter exatamente {n} elementos.")
                    return
                A.append(linha)
            
            print("\nDigite os termos independentes do vetor b (separados por espaço):")
            b = list(map(float, input(f"Vetor b (deve conter {n} números): ").split()))
            if len(b) != n:
                print(f"Erro: O vetor b deve conter exatamente {n} elementos.")
                return
        except ValueError:
            print("Erro de Entrada: Favor digitar valores numéricos reais válidos.")
            return
    else:
        A = A_padrao
        b = b_padrao
        n = 3

    # Se houver elementos nulos na diagonal principal, tenta reordenar antes de fazer a análise
    if any(abs(A[i][i]) < 1e-12 for i in range(n)):
        print("\n[AVISO] Elemento nulo detectado na diagonal principal. Tentando reordenar o sistema...")
        try:
            A, b = reordenar_sistema(A, b)
            print("[OK] Sistema reordenado com sucesso para remover zeros da diagonal!")
        except ValueError as err:
            print(f"\n[ERRO] {err}")
            return

    # Análise matemática de convergência
    diag_dominante = check_diagonal_dominance(A)
    sassenfeld_ok, beta_max = check_sassenfeld(A)

    print("\n--- Análise Matemática de Convergência ---")
    if diag_dominante:
        print("[OK] O sistema é ESTRITAMENTE DIAGONAL DOMINANTE. Convergência garantida.")
    else:
        print("[AVISO] O sistema NÃO é estritamente diagonal dominante.")
        
    if sassenfeld_ok:
        print(f"[OK] O sistema atende ao critério de Sassenfeld (Beta máximo = {beta_max:.4f} < 1.0). Convergência garantida.")
    else:
        print(f"[ALERTA] O sistema NÃO atende ao critério de Sassenfeld (Beta máximo = {beta_max:.4f} >= 1.0).")

    if not diag_dominante and not sassenfeld_ok:
        print("\n================================================================================")
        print(" ATENÇÃO: NENHUM CRITÉRIO DE CONVERGÊNCIA FOI SATISFEITO!")
        print(" O método de Gauss-Seidel pode divergir e não encontrar uma resposta válida.")
        print("================================================================================")
        confirmacao = input("Deseja forçar a execução mesmo assim? [S/N]: ").strip().upper()
        if confirmacao != 'S':
            print("Execução abortada pelo usuário.")
            return

    # Coleta de parâmetros interativos adicionais
    print("\n--- Configuração de Parâmetros e Chutes ---")
    try:
        x0 = []
        for i in range(n):
            val = float(input(f"Digite o chute inicial para x{i+1}: "))
            x0.append(val)
    except ValueError:
        print("Erro: O chute inicial deve ser composto por números reais.")
        return

    try:
        Es = float(input("Digite a tolerância de erro de parada (Es em %): "))
        if Es <= 0:
            print("Erro: A tolerância de erro deve ser maior que zero.")
            return
    except ValueError:
        print("Erro: A tolerância deve ser um valor numérico.")
        return

    try:
        N0 = int(input("Digite o limite máximo de iterações de segurança (N0): "))
        if N0 <= 0:
            print("Erro: O limite de iterações deve ser maior que zero.")
            return
    except ValueError:
        print("Erro: O limite de iterações deve ser um número inteiro.")
        return

    print("\nProcessando iterações do Método de Gauss-Seidel...")
    iteracoes, sucesso, x_final, mensagem = metodo_gauss_seidel(A, b, x0, Es, N0)

    if len(iteracoes) == 0:
        print(f"\nErro fatal durante processamento: {mensagem}")
        return

    # Apresenta tabela de iterações no terminal
    print("\nTabela de Convergência Iterativa (Gauss-Seidel):")
    table_print = []
    for it in iteracoes:
        linha = [it['Iteracao']]
        # Adiciona valores estimados
        for v in it['x']:
            linha.append(f"{v:.6f}")
        # Adiciona o maior erro aproximado
        linha.append(f"{it['Erro_Maximo']:.6f}%")
        table_print.append(linha)
    
    headers_print = ["Iter."]
    for i in range(n):
        headers_print.append(f"x{i+1}")
    headers_print.append("Erro Máx")

    print(tabulate(table_print, headers=headers_print, tablefmt="fancy_grid"))

    print(f"\nResultado da Execução: {mensagem}")
    sol_str = ", ".join(f"x{i+1} = {x_final[i]:.6f}" for i in range(n))
    print(f"Solução obtida: [{sol_str}]")

    # Escrita de logs e geração gráfica
    try:
        salvar_log(A, b, x0, Es, N0, iteracoes, sucesso, x_final, mensagem)
        print("-> Tabela detalhada e resumo salvos em 'resultados_gauss_seidel.txt'.")
    except Exception as err:
        print(f"Erro ao salvar arquivo de texto 'resultados_gauss_seidel.txt': {err}")

    try:
        gerar_graficos(iteracoes, n, x0)
        print("-> Gráfico duplo contendo o histórico de convergência salvo em 'grafico_gauss_seidel.png'.")
    except Exception as err:
        print(f"Erro ao gerar/salvar imagem 'grafico_gauss_seidel.png': {err}")

    print("================================================================================")

if __name__ == "__main__":
    main()
