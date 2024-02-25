#include <iostream>
#include <vector>
#include <cmath>

// Função de ativação (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Função para imprimir a ativação dos neurônios em uma matriz
void printActivation(const std::vector<std::vector<double>>& activation) {
    std::cout << "Ativação dos neurônios:" << std::endl;
    for (const auto& row : activation) {
        for (double val : row) {
            std::cout << (val > 0.5 ? "■" : "□") << " ";
        }
        std::cout << std::endl;
    }
}

// Função para exibir o tabuleiro do jogo da velha
void printBoard(const std::vector<char>& board) {
    std::cout << "  " << board[0] << " | " << board[1] << " | " << board[2] << std::endl;
    std::cout << " -----------" << std::endl;
    std::cout << "  " << board[3] << " | " << board[4] << " | " << board[5] << std::endl;
    std::cout << " -----------" << std::endl;
    std::cout << "  " << board[6] << " | " << board[7] << " | " << board[8] << std::endl;
}

// Classe da Rede Neural
class RedeNeural {
private:
    int tamEntrada;
    int tamOculta;
    int tamSaida;
    double taxaAprendizado;
    std::vector<std::vector<double>> pesosEntradaOculta;
    std::vector<std::vector<double>> pesosOcultaSaida;

public:
    RedeNeural(int tamEntrada, int tamOculta, int tamSaida, double taxaAprendizado) {
        this->tamEntrada = tamEntrada;
        this->tamOculta = tamOculta;
        this->tamSaida = tamSaida;
        this->taxaAprendizado = taxaAprendizado;

        // Inicialização dos pesos
        pesosEntradaOculta.resize(tamEntrada, std::vector<double>(tamOculta, 0.0));
        pesosOcultaSaida.resize(tamOculta, std::vector<double>(tamSaida, 0.0));

        // Inicialização dos pesos com valores aleatórios
        for (int i = 0; i < tamEntrada; ++i) {
            for (int j = 0; j < tamOculta; ++j) {
                pesosEntradaOculta[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Valores entre -1 e 1
            }
        }

        for (int i = 0; i < tamOculta; ++i) {
            for (int j = 0; j < tamSaida; ++j) {
                pesosOcultaSaida[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Valores entre -1 e 1
            }
        }
    }

    // Feedforward
    std::vector<double> forward(std::vector<double> entradas, std::vector<std::vector<double>>& ativacaoOculta) {
        std::vector<double> oculta(tamOculta, 0.0);
        std::vector<double> saida(tamSaida, 0.0);

        // Cálculo da camada oculta
        for (int i = 0; i < tamOculta; ++i) {
            double soma = 0.0;
            for (int j = 0; j < tamEntrada; ++j) {
                soma += entradas[j] * pesosEntradaOculta[j][i];
            }
            oculta[i] = sigmoid(soma);
        }
        ativacaoOculta.push_back(oculta);

        // Cálculo da camada de saída
        for (int i = 0; i < tamSaida; ++i) {
            double soma = 0.0;
            for (int j = 0; j < tamOculta; ++j) {
                soma += oculta[j] * pesosOcultaSaida[j][i];
            }
            saida[i] = sigmoid(soma);
        }

        return saida;
    }

    // Treinamento com backpropagation
    void train(std::vector<double> entradas, std::vector<double> alvos) {
        // Feedforward
        std::vector<std::vector<double>> ativacaoOculta;
        std::vector<double> saida = forward(entradas, ativacaoOculta);

        // Backpropagation
        std::vector<double> errosSaida(tamSaida, 0.0);
        for (int i = 0; i < tamSaida; ++i) {
            errosSaida[i] = (alvos[i] - saida[i]) * saida[i] * (1 - saida[i]);
        }

        std::vector<double> errosOculta(tamOculta, 0.0);
        for (int i = 0; i < tamOculta; ++i) {
            double erro = 0.0;
            for (int j = 0; j < tamSaida; ++j) {
                erro += errosSaida[j] * pesosOcultaSaida[i][j];
            }
            errosOculta[i] = erro * ativacaoOculta.back()[i] * (1 - ativacaoOculta.back()[i]);
        }

        // Atualização dos pesos
        for (int i = 0; i < tamEntrada; ++i) {
            for (int j = 0; j < tamOculta; ++j) {
                pesosEntradaOculta[i][j] += taxaAprendizado * errosOculta[j] * entradas[i];
            }
        }

        for (int i = 0; i < tamOculta; ++i) {
            for (int j = 0; j < tamSaida; ++j) {
                pesosOcultaSaida[i][j] += taxaAprendizado * errosSaida[j] * ativacaoOculta.back()[i];
            }
        }
    }
};

int main() {
    // Exemplo de uso da rede neural
    int tamEntrada = 9; // Tamanho do tabuleiro de jogo (3x3)
    int tamOculta = 9; // Número de neurônios na camada oculta
    int tamSaida = 9; // Número de possíveis movimentos
    double taxaAprendizado = 0.1; // Taxa de aprendizado

    RedeNeural nn(tamEntrada, tamOculta, tamSaida, taxaAprendizado);

    // Tabuleiro de jogo da velha
    std::vector<char> tabuleiro = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};

    // Exemplo de entrada e saída para o treinamento
    std::vector<double> entradas = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // Tabuleiro de jogo
    std::vector<double> alvos = {0, 0, 0, 0, 0, 1, 0, 0, 0}; // Movimento desejado

    // Treinamento da rede neural
    nn.train(entradas, alvos);

    // Mostra o tabuleiro inicial
    std::cout << "Tabuleiro inicial:" << std::endl;
    printBoard(tabuleiro);

    // Teste da rede neural
    std::vector<std::vector<double>> ativacaoOculta;
    std::vector<double> testeEntradas = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // Novo tabuleiro de jogo
    std::vector<double> saidas = nn.forward(testeEntradas, ativacaoOculta); // Calcula os movimentos possíveis
    // Mostra os movimentos possíveis
    std::cout << "\nMovimentos possíveis:" << std::endl;
    for (int i = 0; i < tamSaida; ++i) {
        std::cout << "Movimento " << i << ": " << saidas[i] << std::endl;
    }

    // Mostra a ativação dos neurônios na camada oculta
    printActivation(ativacaoOculta);

    return 0;
}
