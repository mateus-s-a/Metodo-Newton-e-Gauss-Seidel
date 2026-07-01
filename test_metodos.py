#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suíte de Testes Unitários para os Métodos de Newton-Raphson e Gauss-Seidel.
Contém 10 testes unitários para cada método, cobrindo cenários de sucesso,
limites, erros e convergência difícil.
"""

import unittest
import numpy as np

# Importa as funções sob teste
import metodo_newton
import metodo_gauss_seidel

# Armazena as funções originais de Newton para restauração após os testes
ORIGINAL_F = metodo_newton.f
ORIGINAL_DF = metodo_newton.df


class TestMetodoNewton(unittest.TestCase):

    def setUp(self):
        # Restaura as funções padrão antes de cada teste
        metodo_newton.f = ORIGINAL_F
        metodo_newton.df = ORIGINAL_DF

    def tearDown(self):
        # Restaura as funções padrão depois de cada teste
        metodo_newton.f = ORIGINAL_F
        metodo_newton.df = ORIGINAL_DF

    def test_newton_typical_convergence(self):
        """1. Caso típico de convergência para f(x) = x^3 - x - 2."""
        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=1.0, Es=1e-6, N0=100
        )
        self.assertTrue(sucesso)
        self.assertAlmostEqual(raiz, 1.52137971, places=6)
        self.assertGreater(len(iteracoes), 0)

    def test_newton_zero_derivative(self):
        """2. Derivada nula no chute inicial (f'(x0) = 0)."""
        # Para f(x) = x^3 - x - 2, f'(x) = 3x^2 - 1. A derivada é nula em x = sqrt(1/3) ~ 0.57735.
        x0 = (1.0 / 3.0) ** 0.5
        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=x0, Es=1e-6, N0=5
        )
        self.assertFalse(sucesso)
        self.assertIn("derivada nula", msg.lower() or "divisão por zero" in msg.lower())

    def test_newton_infinite_cycle(self):
        """3. Oscilação infinita/ciclo periódico (não convergente)."""
        # Definindo uma função personalizada que oscila entre -1 e 1 para x0 = 1.0
        # f(x) = x^3 - 5x
        # f'(x) = 3x^2 - 5
        def custom_f(x):
            return x**3 - 5*x

        def custom_df(x):
            return 3*x**2 - 5

        metodo_newton.f = custom_f
        metodo_newton.df = custom_df

        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=1.0, Es=1e-6, N0=5
        )
        self.assertFalse(sucesso)
        self.assertIn("limite", msg.lower())
        # Verifica se oscilou entre 1 e -1
        self.assertAlmostEqual(iteracoes[0]['xi'], 1.0)
        self.assertAlmostEqual(iteracoes[0]['xi_mais_1'], -1.0)
        self.assertAlmostEqual(iteracoes[1]['xi'], -1.0)
        self.assertAlmostEqual(iteracoes[1]['xi_mais_1'], 1.0)

    def test_newton_zero_convergence_bug_fix(self):
        """4. Validação da correção do bug onde x_novo = 0.0 causava falsa convergência rápida."""
        # Se x_novo for exatamente 0.0, mas não for a raiz, o erro relativo deve ser calculado 
        # corretamente e o algoritmo deve continuar iterando em vez de alegar sucesso imediato.
        # Chute x0 = -1.0 leva a x_novo = 0.0 na 1ª iteração para f(x) = x^3 - x - 2
        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=-1.0, Es=1e-6, N0=50
        )
        # Não deve convergir com sucesso no primeiro passo (onde x_novo = 0.0 e f(0) = -2)
        # O método deve continuar até convergir para a raiz correta (~1.5213)
        self.assertTrue(sucesso)
        self.assertAlmostEqual(raiz, 1.52137971, places=6)
        self.assertGreater(len(iteracoes), 1)

    def test_newton_extreme_tolerance(self):
        """5. Tolerância extremamente baixa exige precisão máxima."""
        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=2.0, Es=1e-12, N0=100
        )
        self.assertTrue(sucesso)
        self.assertAlmostEqual(raiz, 1.5213797068, places=8)

    def test_newton_iteration_limit(self):
        """6. Limite estrito de iterações atingido sem convergência."""
        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=3.0, Es=1e-12, N0=3
        )
        self.assertFalse(sucesso)
        self.assertEqual(len(iteracoes), 3)
        self.assertIn("limite", msg.lower())

    def test_newton_multiple_root(self):
        """7. Raiz de multiplicidade superior (convergência linear mais lenta)."""
        # f(x) = (x-1)^2 = x^2 - 2x + 1
        # f'(x) = 2x - 2
        # Raiz em x = 1.0. A convergência é linear e a derivada se aproxima de zero na raiz.
        def custom_f(x):
            return (x - 1.0) ** 2

        def custom_df(x):
            return 2.0 * x - 2.0

        metodo_newton.f = custom_f
        metodo_newton.df = custom_df

        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=1.5, Es=1e-3, N0=100
        )
        # Deve encontrar a raiz, embora demande mais iterações
        self.assertTrue(sucesso)
        self.assertAlmostEqual(raiz, 1.0, places=2)

    def test_newton_divergence(self):
        """8. Função divergente (erro cresce a cada passo)."""
        # f(x) = x^(1/3)
        # f'(x) = 1 / (3 * x^(2/3))
        # Para qualquer x0 != 0, x_novo = x0 - x0^(1/3) * 3 * x0^(2/3) = -2 * x0 (diverge).
        def custom_f(x):
            return np.sign(x) * (abs(x) ** (1.0 / 3.0))

        def custom_df(x):
            return (1.0 / 3.0) * (abs(x) ** (-2.0 / 3.0))

        metodo_newton.f = custom_f
        metodo_newton.df = custom_df

        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=0.5, Es=1e-6, N0=10
        )
        self.assertFalse(sucesso)
        self.assertEqual(len(iteracoes), 10)
        # O último x deve ser (-2)^10 * 0.5 = 512
        self.assertAlmostEqual(abs(raiz), 512.0)

    def test_newton_constant_function(self):
        """9. Função constante (sem raiz, derivada sempre nula)."""
        def custom_f(x):
            return 5.0

        def custom_df(x):
            return 0.0

        metodo_newton.f = custom_f
        metodo_newton.df = custom_df

        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=1.0, Es=1e-6, N0=5
        )
        self.assertFalse(sucesso)
        self.assertIn("derivada nula", msg.lower())

    def test_newton_single_iteration(self):
        """10. Comportamento com apenas 1 iteração de limite."""
        iteracoes, sucesso, raiz, msg = metodo_newton.metodo_newton_raphson(
            x0=2.0, Es=1e-6, N0=1
        )
        self.assertEqual(len(iteracoes), 1)


class TestMetodoGaussSeidel(unittest.TestCase):

    def test_gs_typical_3x3(self):
        """1. Sistema padrão 3x3 diagonal dominante (convergência garantida)."""
        A = [
            [10.0, 2.0, -1.0],
            [-3.0, -6.0, 2.0],
            [1.0, 1.0, 5.0]
        ]
        b = [27.0, -61.5, -21.5]
        x0 = [0.0, 0.0, 0.0]
        
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, x0, Es=1e-5, N0=100
        )
        self.assertTrue(sucesso)
        # Solução exata: x1 = 3.0, x2 = 12.5 (espera, no main está: -61.5 para segunda linha...)
        # Vamos verificar a solução correspondente
        # 10*x1 + 2*x2 - x3 = 27
        # -3*x1 - 6*x2 + 2*x3 = -61.5
        # x1 + x2 + 5*x3 = -21.5
        # Solução esperada: [3.0, 12.5, -7.0] -> na verdade, vamos testar se Ax = b com tolerancia
        np_A = np.array(A)
        np_x = np.array(x_final)
        np_b = np.array(b)
        self.assertTrue(np.allclose(np_A.dot(np_x), np_b, atol=1e-3))

    def test_gs_sassenfeld_only(self):
        """2. Matriz que não é diagonal dominante mas atende a Sassenfeld."""
        # Matriz A = [[2, 1, 0], [3, 4, 2], [0, 1, 2]]
        # Não é diag dominante porque na linha 2: |4| <= |3| + |2| = 5.
        # Mas atende a Sassenfeld (beta_max = 0.875 < 1.0).
        A = [
            [2.0, 1.0, 0.0],
            [3.0, 4.0, 2.0],
            [0.0, 1.0, 2.0]
        ]
        b = [5.0, 17.0, 5.0]  # Solução exata: [1, 3, 1]
        x0 = [0.0, 0.0, 0.0]
        
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, x0, Es=1e-4, N0=100
        )
        self.assertTrue(sucesso)
        self.assertAlmostEqual(x_final[0], 1.0, places=3)
        self.assertAlmostEqual(x_final[1], 3.0, places=3)
        self.assertAlmostEqual(x_final[2], 1.0, places=3)

    def test_gs_zero_convergence_bug_fix(self):
        """3. Correção do bug onde xi se tornava 0.0 e gerava falsa convergência rápida."""
        # Se uma das variáveis do sistema se aproxima de 0.0, o erro relativo não deve ser zero,
        # a menos que a iteração anterior também tenha sido próxima de 0.0.
        A = [
            [5.0, 1.0],
            [1.0, 5.0]
        ]
        b = [0.0, 5.0]  # Solução: x1 = -5/24 ~ -0.2083, x2 = 25/24 ~ 1.0416
        # Chute inicial em que x1_novo se torna 0.0
        # Na 1ª iteração: x1 = (0.0 - 1.0 * x2) / 5.0 = -x2 / 5.0.
        # Se escolhermos x2 = 0.0 e chute inicial x0 = [-100.0, 0.0]
        # x1_novo = (0.0 - 0.0)/5.0 = 0.0.
        # Anteriormente, x1_velho = -100.0. A variação foi grande, mas como x1_novo = 0.0,
        # o bug definia erro relativo como 0.0%.
        x0 = [-100.0, 0.0]
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, x0, Es=1e-5, N0=100
        )
        self.assertTrue(sucesso)
        self.assertAlmostEqual(x_final[0], -0.20833333, places=5)
        self.assertAlmostEqual(x_final[1], 1.04166667, places=5)
        self.assertGreater(len(iteracoes), 1)

    def test_gs_requires_reordering(self):
        """4. Elemento nulo na diagonal reordenável com sucesso."""
        A = [
            [0.0, 2.0],
            [3.0, 1.0]
        ]
        b = [4.0, 5.0]  # Solução: x1 = 1, x2 = 2.
        # Deve reordenar internamente para [[3, 1], [0, 2]] e b para [5, 4]
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, [0.0, 0.0], Es=1e-5, N0=50
        )
        self.assertTrue(sucesso)
        self.assertAlmostEqual(x_final[0], 1.0, places=4)
        self.assertAlmostEqual(x_final[1], 2.0, places=4)

    def test_gs_impossible_reordering(self):
        """5. Elemento nulo na diagonal impossível de reordenar."""
        A = [
            [0.0, 1.0],
            [0.0, 2.0]
        ]
        b = [3.0, 4.0]
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, [0.0, 0.0], Es=1e-5, N0=10
        )
        self.assertFalse(sucesso)
        self.assertIn("reordenar", msg.lower())

    def test_gs_divergent_system(self):
        """6. Sistema divergente que atinge o limite de iterações."""
        A = [
            [1.0, 3.0],
            [3.0, 1.0]
        ]
        b = [4.0, 4.0]  # Diagonal fraca, diverge.
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, [0.0, 0.0], Es=1e-5, N0=10
        )
        self.assertFalse(sucesso)
        self.assertEqual(len(iteracoes), 10)
        self.assertIn("limite", msg.lower())

    def test_gs_large_system(self):
        """7. Sistema de grande dimensão (10x10) diagonal dominante."""
        n = 10
        # Cria matriz diagonal dominante
        A = []
        b = []
        for i in range(n):
            linha = [1.0] * n
            linha[i] = 20.0  # Dominância diagonal forte
            A.append(linha)
            b.append(20.0 + (n - 1))  # Solução exata: todos x_i = 1.0
            
        x0 = [0.0] * n
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, x0, Es=1e-5, N0=100
        )
        self.assertTrue(sucesso)
        for i in range(n):
            self.assertAlmostEqual(x_final[i], 1.0, places=4)

    def test_gs_singular_system(self):
        """8. Sistema singular inconsistente (sem solução)."""
        A = [
            [1.0, 2.0],
            [2.0, 4.0]
        ]
        b = [3.0, 7.0]  # Sistema incompatível/inconsistente
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, [0.0, 0.0], Es=1e-5, N0=10
        )
        self.assertFalse(sucesso)

    def test_gs_incompatible_dimensions(self):
        """9. Tratamento de erro para dimensões incompatíveis de parâmetros."""
        A = [
            [2.0, 1.0],
            [1.0, 2.0]
        ]
        b = [3.0]  # A é 2x2, mas b é tamanho 1
        x0 = [0.0, 0.0]
        
        with self.assertRaises(ValueError):
            metodo_gauss_seidel.metodo_gauss_seidel(A, b, x0, Es=1e-5, N0=10)

    def test_gs_limit_iterations(self):
        """10. Comportamento com apenas 1 iteração de limite."""
        A = [
            [10.0, 1.0],
            [1.0, 10.0]
        ]
        b = [11.0, 11.0]
        iteracoes, sucesso, x_final, msg = metodo_gauss_seidel.metodo_gauss_seidel(
            A, b, [0.0, 0.0], Es=1e-5, N0=1
        )
        self.assertEqual(len(iteracoes), 1)


if __name__ == "__main__":
    unittest.main()
