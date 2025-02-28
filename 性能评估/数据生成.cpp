#include <cmath>
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <unordered_map>
#include <limits>
#include <random>
#include <chrono>
#include <stdexcept>

using namespace std;

// �������ӿ�
class Encoder {
public:
    virtual std::vector<int> encode(const std::vector<int>& message) = 0;
    virtual size_t get_block_size() const = 0;
    virtual ~Encoder() = default;
};

// ���Է����������
class LinearBlockCodeEncoder : public Encoder {
public:

    LinearBlockCodeEncoder(int k, int n,vector<vector<int>> generatorMatrix) 
        : k(k), n(n),generatorMatrix(generatorMatrix) {
        if (!validateBlockCode(k, n)) {
            throw std::invalid_argument("n ������� k");
        }
    }
    std::vector<int> encode(const std::vector<int>& input) override {
        vector<vector<int>> result;
        for (size_t i = 0; i < input.size(); i += k) {
            // ��ԭʼ�����н�ȡ����Ϊ n ����������ע�����һ�ο��ܲ��� n
            std::vector<int> segment(input.begin() + i, input.begin() + std::min(i + k, input.size()));
            result.push_back(segment);
        }
        vector<int> output;
        for(const auto& inputs : result){
            vector<int> encode=encode_block(inputs);
            output.insert(output.end(), encode.begin(), encode.end());
        }
    return output;
    }

    size_t get_block_size() const override {
        return k;
    }

private:
    vector<vector<int>> generatorMatrix;
    
    int k;
    int n;

    std::vector<int> encode_block(const std::vector<int>& block) {
        vector<int> output(n, 0);

        // ���Է�������룺��������غ����ɾ������
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                output[j] ^= block[i] * generatorMatrix[i][j];  // ģ2�ӷ�
            }
        }
        return output;
    }
    // �жϾ����Ƿ���k��n��
    bool validateBlockCode(int k, int n) {
        return (n > k);  // n ������� k
    }

};

// ����������
class ConvolutionalCodeEncoder : public Encoder {
private:
    int n; // ���������
    int k; // ���������
    int m; // �Ĵ�������

    // ���ɶ���ʽ���󣬴�СΪ n x (m+1) x k
    vector<vector<vector<int>>> generatorMatrix;

    // �Ĵ������飬���ڴ洢 k-bit ��������
    vector<vector<int>> shiftRegister;

public:
    // ���캯��
    ConvolutionalCodeEncoder(int n, int k, int m, vector<vector<vector<int>>> generatorMatrix)
        : n(n), k(k), m(m), generatorMatrix(generatorMatrix) {
        shiftRegister = vector<vector<int>>(m + 1, vector<int>(k, 0));
    }

    // ���¼Ĵ���������
    void shiftRegisters(const vector<int>& inputBits) {
        // �ƶ��Ĵ�������
        for (int i = m; i > 0; i--) {
            shiftRegister[i] = shiftRegister[i - 1];
        }
        // ����������һ���Ĵ���
        for (int i = 0; i < k; i++) {
            shiftRegister[0][i] = inputBits[i];
        }
    }

    // �������ɶ���ʽ�������
    vector<int> encode2(const vector<int>& inputBits){
        vector<int> outputBits(n, 0);

        // ���¼Ĵ�������
        shiftRegisters(inputBits);

        // �����������
        for (int i = 0; i < n; i++) {          // ����ÿ���������
            for (int j = 0; j <= m; j++) {      // ����ÿ���Ĵ���
                for (int l = 0; l < k; l++) {   // ���ڼĴ�����ÿ������
                    outputBits[i] ^= shiftRegister[j][l] & generatorMatrix[i][j][l];
                }
            }
        }

        return outputBits;
    }

    // ���������������У�������β������ֹ
        vector<int> encode(const vector<int>& inputBitsSequence) override {
        vector<int> encodedSequence;
        std::vector<std::vector<int>> result;
        // ��������һά����
        for (size_t i = 0; i < inputBitsSequence.size(); i += k) {
            // ��ԭʼ�����н�ȡ����Ϊ n ����������ע�����һ�ο��ܲ��� n
            std::vector<int> segment(inputBitsSequence.begin() + i, inputBitsSequence.begin() + std::min(i + k, inputBitsSequence.size()));
            result.push_back(segment);
        }
        for (const auto& inputBits : result) {
            vector<int> outputBits = encode2(inputBits);
            encodedSequence.insert(encodedSequence.end(), outputBits.begin(), outputBits.end());
        }
        // // ����β������ֹ
        // vector<int> zeroBits(k, 0);
        // for (int i = 0; i < m; i++) {
        //     encodedSequence.push_back(encode(zeroBits));
        // }
        return encodedSequence;
    }

    // ʵ�� get_block_size ����
    size_t get_block_size() const override {
        return k; // ���������������Ϊ���С
    }
};

// ģ���Ԫ�Գ��ŵ�
std::vector<int> simulate_bsc(const std::vector<int>& codeword, double error_probability) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::vector<int> transmitted_codeword = codeword;

    for (int& bit : transmitted_codeword) {
        if (dis(gen) < error_probability) {
            bit ^= 1; // ��תλ
        }
    }

    return transmitted_codeword;
}

// ���ַ���ת��Ϊ����������
std::vector<int> string_to_binary(const std::string& str) {
    std::vector<int> binary;
    for (char c : str) {
        for (int i = 7; i >= 0; --i) {
            binary.push_back((c >> i) & 1);
        }
    }
    return binary;
}

// ���ļ�����ת��Ϊ����������
std::vector<int> file_to_binary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<int> binary;
    char byte;
    while (file.get(byte)) {
        for (int i = 7; i >= 0; --i) {
            binary.push_back((byte >> i) & 1);
        }
    }

    return binary;
}


// �Զ����ϣ����
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// �Զ�����ȱȽϺ���
struct VectorEqual {
    bool operator()(const vector<int>& lhs, const vector<int>& rhs) const {
        return lhs == rhs;
    }
};


// �������ӿ�
class Decoder {
public:
    virtual std::vector<int> decode(const std::vector<int>& received) = 0;
    virtual ~Decoder() = default;
};

// ���Է���������� //������ķ����
class LinearBlockCodeDecoder : public Decoder {
public:
    LinearBlockCodeDecoder(int k, int n, const vector<vector<int>>& generatorMatrix) 
        : k(k), n(n), generatorMatrix(generatorMatrix) {
        if (!validateBlockCode(k, n)) {
            throw std::invalid_argument("n ������� k");
        }
    }
    vector<int> decode(const vector<int>& receivedBits) override
     {
        vector<vector<int>> H = generateCheckMatrix(generatorMatrix, k, n);
        unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual> syndromeTable = generateSyndromeTable(H, n);
        std::vector<std::vector<int>> received;
    
        // ��������һά����
        for (size_t i = 0; i < receivedBits.size(); i += n) {
            // ��ԭʼ�����н�ȡ����Ϊ n ����������ע�����һ�ο��ܲ��� n
            std::vector<int> segment(receivedBits.begin() + i, receivedBits.begin() + std::min(i + n, receivedBits.size()));
            received.push_back(segment);
        }
        std::vector<int> decoded_data;
        for(int i=0;i<received.size();i++){
            vector<int> decode= Sdecode(received[i],H,syndromeTable);
            decoded_data.insert(decoded_data.end(),decode.begin(),decode.begin()+k);
        }
        return decoded_data;
    }

    // ������ķ����
    vector<int> Sdecode(const vector<int>& received, const vector<vector<int>>& H, const unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual>& syndromeTable) {
        
        // ����У����
        vector<int> syndrome = calculateSyndrome(received, H);

        // ���Ҵ���ģʽ
        vector<int> errorPattern = findErrorPattern(syndrome, syndromeTable);

        // ���û���ҵ�����ģʽ����Ϊû�д���
        if (errorPattern.empty()) {
            return received;
        }

        // �������󣺽������������ģʽ����ģ2�ӷ�
        vector<int> corrected(received.size());
        for (size_t i = 0; i < received.size(); ++i) {
            corrected[i] = received[i] ^ errorPattern[i];
        }

        return corrected;
    }

private:
    int n;
    int k;
    vector<vector<int>> generatorMatrix;

    // �жϾ����Ƿ���k��n��
    bool validateBlockCode(int k, int n) {
        return (n > k);  // n ������� k
    }

    // ����У����� H = (P^T | I)
    vector<vector<int>> generateCheckMatrix(const vector<vector<int>>& G, int k, int n) {
        int rows = n - k; // H ������
        int cols = n;     // H ������
        vector<vector<int>> H(rows, vector<int>(cols, 0));

        // ��ȡ���ɾ��� G �е��Ҳಿ�� P
        vector<vector<int>> P(k, vector<int>(rows, 0));
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < rows; ++j) {
                P[i][j] = G[i][k + j];  // ��ȡ G ���Ҳಿ�� P
            }
        }

        // ת�� P �������� H �����
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < k; ++j) {
                H[i][j] = P[j][i];  // P ��ת��
            }
        }

        // ����У�������Ҳ൥λ���� I
        for (int i = 0; i < rows; ++i) {
            H[i][k + i] = 1;  // �� H ���Ҳ���ӵ�λ���󲿷�
        }

        return H;
    }


    // ����У������ת��
    vector<vector<int>> transposeMatrix(const vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<vector<int>> transposed(cols, vector<int>(rows));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    // ����У���� S = r * H^T
    vector<int> calculateSyndrome(const vector<int>& received, const vector<vector<int>>& H) {
        int n = H[0].size();  // H �������������ֵĳ��ȣ�
        int m = H.size();     // H ��������У�鷽�̵�������

        // ��ȡ H ��ת�þ��� H^T
        vector<vector<int>> HT = transposeMatrix(H);

        // ����У���� S = r * H^T
        vector<int> syndrome(m, 0);  // ��ʼ��У����

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                syndrome[i] ^= received[j] * HT[i][j];  // ģ2�ӷ�
            }
        }
        
        return syndrome;
    }


    // ���Ҵ���ģʽ������У���ӷ��ض�Ӧ�Ĵ���ģʽ
    vector<int> findErrorPattern(const vector<int>& syndrome, const unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual>& syndromeTable) {
        auto it = syndromeTable.find(syndrome);
        if (it != syndromeTable.end()) {
            return it->second;  // �ҵ���Ӧ�Ĵ���ģʽ
        }
        return vector<int>();  // û�ҵ�����ģʽ�����ؿ�
    }

    // ����������ķ���ұ�
    unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual> generateSyndromeTable(const vector<vector<int>>& H, int n) {
        unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual> syndromeTable;

        // �������п��ܵĵ����ش���ģʽ
        for (int i = 0; i < n; ++i) {
            // ����һ������ģʽ��ֻ�е�iλΪ1����ʾ��iλ��������
            vector<int> errorPattern(n, 0);
            errorPattern[i] = 1;

            // ����ô���ģʽ��Ӧ��У����
            vector<int> syndrome = calculateSyndrome(errorPattern, H);

            // ��У���������ģʽӳ�䵽����
            syndromeTable[syndrome] = errorPattern;
        }

        return syndromeTable;
    }
};


class ConvolutionalCodeDecoder : public Decoder {
private:
    int n, k, m;
    int numStates;
    vector<vector<vector<int>>> generatorMatrix;

    // ״̬ת�ƺ����
    vector<vector<int>> stateTransition;
    vector<vector<vector<int>>> outputBits;

    // ��ʼ��״̬ת�ƺ����
    void initStateTransition() {
        stateTransition.resize(numStates, vector<int>(1 << k, 0));
        outputBits.resize(numStates, vector<vector<int>>(1 << k, vector<int>(n)));

        for (int state = 0; state < numStates; ++state) {
            for (int input = 0; input < (1 << k); ++input) {
                int newState = ((input << ((m-1) * k)) | (state >> k));
                stateTransition[state][input] = newState;
                outputBits[state][input] = calculateOutput(state, input);
            }
        }
    }

    // ������������ʮ����״̬ת��Ϊ�����Ʊ�����ʸ��
    vector<int> decimalToBinary(int state, int bits) {
        vector<int> binary(bits, 0);
        for (int i = 0; i < bits; ++i) {
            binary[bits - 1 - i] = (state >> i) & 1;
        }
        return binary;
    }

    // ���ݵ�ǰ״̬������������
    vector<int> calculateOutput(int state, int input) {
        vector<int> output(n, 0);
        vector<int> reg(k, 0);
        for (int i = 0; i < n; ++i) {
            int outBit = 0;
            for (int j = 0; j < m; ++j) {
                reg = decimalToBinary((state >> ((m - j - 1) * k)) & ((1 << k) - 1), k); // ��ȡ�Ĵ����е�ʸ��״̬
                for (int l = 0; l < k; ++l) {
                    outBit ^= reg[l] & generatorMatrix[i][j+1][l];
                }
            }
            // �Ե�ǰ����ʸ�����д���
            reg = decimalToBinary(input, k);
            for (int l = 0; l < k; ++l) {
                outBit ^= reg[l] & generatorMatrix[i][0][l];
            }
            output[i] = outBit;
        }
        return output;
    }

    // ����·���ĺ�������
    int hammingDistance(const vector<int>& a, const vector<int>& b) {
        int distance = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) {
                ++distance;
            }
        }
        return distance;
    }

public:
    ConvolutionalCodeDecoder(int n, int k, int m, const vector<vector<vector<int>>>& generatorMatrix)
        : n(n), k(k), m(m), generatorMatrix(generatorMatrix) {
        numStates = 1 << (k * m); // ����״̬��
        initStateTransition();
    }

    // Viterbi �������
        vector<int> decode(const vector<int>& receivedBits) override {
        std::vector<std::vector<int>> received;
    
        // ��������һά����
        for (size_t i = 0; i < receivedBits.size(); i += n) {
            // ��ԭʼ�����н�ȡ����Ϊ n ����������ע�����һ�ο��ܲ��� n
            std::vector<int> segment(receivedBits.begin() + i, receivedBits.begin() + std::min(i + n, receivedBits.size()));
            received.push_back(segment);
        }
        int T = received.size();
        vector<vector<int>> pathMetric(2, vector<int>(numStates, numeric_limits<int>::max()));
        vector<vector<int>> prevState(numStates, vector<int>(T, -1));

        pathMetric[0][0] = 0; // ��ʼ״̬

        for (int t = 0; t < T; ++t) {
            for (int state = 0; state < numStates; ++state) {
                if (pathMetric[t % 2][state] == numeric_limits<int>::max()) continue;

                for (int input = 0; input < (1 << k); ++input) {
                    int newState = stateTransition[state][input];
                    vector<int> output = outputBits[state][input];

                    int metric = pathMetric[t % 2][state] + hammingDistance(received[t], output);

                    if (metric < pathMetric[(t + 1) % 2][newState]) {
                        pathMetric[(t + 1) % 2][newState] = metric;
                        prevState[newState][t] = state;
                    }
                }
            }

            fill(pathMetric[t % 2].begin(), pathMetric[t % 2].end(), numeric_limits<int>::max());
        }

        int state = min_element(pathMetric[T % 2].begin(), pathMetric[T % 2].end()) - pathMetric[T % 2].begin();
        vector<int> decodedBits;
        for (int t = T - 1; t >= 0; --t) {
            for (int input = 0; input < (1 << k); ++input) {
                if (stateTransition[prevState[state][t]][input] == state) {
                    vector<int>temp = decimalToBinary(input, k);
                    reverse(temp.begin(), temp.end());
                    decodedBits.insert(decodedBits.end(), temp.begin(), temp.end());
                    break;
                }
            }
            state = prevState[state][t];
        }
        reverse(decodedBits.begin(), decodedBits.end());
        return decodedBits;
    }
};

// ������������ת��Ϊ�ַ���
std::string binary_to_string(const std::vector<int>& binary) {
    std::string str;
    for (size_t i = 0; i < binary.size(); i += 8) {
        char c = 0;
        for (int j = 0; j < 8; ++j) {
            c |= binary[i + j] << (7 - j);
        }
        str += c;
    }
    return str;
}

// ������������д���ļ�
void binary_to_file(const std::vector<int>& binary, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    for (size_t i = 0; i < binary.size(); i += 8) {
        char c = 0;
        for (int j = 0; j < 8; ++j) {
            c |= binary[i + j] << (7 - j);
        }
        file.put(c);
    }
}


// ������ɾ��������ɾ���G��n��m+1�У�ÿ��Ԫ����һ��kάʸ����
vector<vector<vector<int>>> generateConvolutionalMatrix(int n, int k, int m,unsigned int seed) {
    vector<vector<vector<int>>> G(n, vector<vector<int>>(m+1, vector<int>(k)));
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, 1);

    // ������ɶ���ʽ����G
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= m; ++j) {
            for (int l = 0; l < k; ++l) {
                G[i][j][l] = dis(gen);
            }
        }
    }

    return G;
}

// �������е������޹��ԣ�ͨ���������ķ���
bool isLinearlyIndependent(const vector<vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<int>> tempMatrix = matrix;

    for (int col = 0, row = 0; col < cols && row < rows; ++col) {
        int selectedRow = row;
        // �ҵ���һ�������в�Ϊ�����
        for (int i = row; i < rows; ++i) {
            if (tempMatrix[i][col] == 1) {
                selectedRow = i;
                break;
            }
        }

        if (tempMatrix[selectedRow][col] == 0) {
            continue; // �������Ϊ0����������
        }

        // ������ǰ�к� selectedRow ��
        swap(tempMatrix[row], tempMatrix[selectedRow]);

        // ���·���������ȥ��ȷ��ÿ�еĸ���ֻ��һ�� 1
        for (int i = row + 1; i < rows; ++i) {
            if (tempMatrix[i][col] == 1) {
                for (int j = col; j < cols; ++j) {
                    tempMatrix[i][j] ^= tempMatrix[row][j];
                }
            }
        }

        ++row; // ������һ��
    }

    // ������������������������޹�
    for (int i = 0; i < rows; ++i) {
        bool allZero = true;
        for (int j = 0; j < cols; ++j) {
            if (tempMatrix[i][j] != 0) {
                allZero = false;
                break;
            }
        }
        if (allZero) return false;
    }

    return true;
}

// ����������ɾ��� (n, k) ���Է�����
vector<vector<int>> generateGeneratorMatrix(int k, int n, unsigned int seed) {
    // ��ʼ�����ɾ��� G����СΪ k x n
    vector<vector<int>> G(k, vector<int>(n, 0));

    // ��ಿ��Ϊ��λ���� I
    for (int i = 0; i < k; ++i) {
        G[i][i] = 1;
    }

    // �Ҳಿ��Ϊ������� P (��СΪ k x (n - k))����ȷ�������޹�
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < k; ++i) {
        vector<int> tempRow(n - k, 0);
        do {
            for (int j = 0; j < n - k; ++j) {
                tempRow[j] = dis(gen); // ������� 0 �� 1
            }

            // ������ʱ�е�������Ҳ�
            for (int j = k; j < n; ++j) {
                G[i][j] = tempRow[j - k];
            }
        } while (!isLinearlyIndependent(G)); // �������ʱ����������
    }

    return G;
}
// ��Ԫ�Գ��ŵ�����
vector<int> transmitThroughBSC(const vector<int>& codeword, double epsilon, mt19937& gen) {
    uniform_real_distribution<> dis(0, 1);
    vector<int> received = codeword;
    for (auto& bit : received) {
        if (dis(gen) < epsilon) {
            bit = 1 - bit; // ���ط�ת
        }
    }
    return received;
}

// �����������
double calculateBER(const vector<int>& original, const vector<int>& decoded) {
    int errors = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] != decoded[i]) {
            errors++;
        }
    }
    return static_cast<double>(errors) / original.size();
}

void simulatePerformance(double epsilon, Encoder& encoder, Decoder& decoder, vector<int>& infoBits, mt19937& gen, double& ber) {
    vector<int> encoded = encoder.encode(infoBits);
    vector<int> received = simulate_bsc(encoded, epsilon);
    vector<int> decoded = decoder.decode(received);
    ber = calculateBER(infoBits, decoded);
}

int main() {
    // ��������
    int L = 1000; // ��Ϣ���س���
    unsigned int seed = 42; // �������
    mt19937 gen(seed);
    
    // ���������Ϣ����
    vector<int> infoBits(L);
    uniform_int_distribution<> dis(0, 1);
    for (int& bit : infoBits) {
        bit = dis(gen);
    }

    // �����������
    Encoder* noEncoder = nullptr;
    Decoder* noDecoder = nullptr;

    int n = 7, k = 4, m = 3;
    vector<vector<int>> GeneratorMatrix = generateGeneratorMatrix(k, n, 13);

    LinearBlockCodeEncoder lbcEncoder(k,n, GeneratorMatrix);
    LinearBlockCodeDecoder lbcDecoder(k,n,GeneratorMatrix);
    
    vector<vector<vector<int>>> ConvolutionalMatrix = generateConvolutionalMatrix(n, k, m,13);
    ConvolutionalCodeEncoder convEncoder(n,k,m,ConvolutionalMatrix);
    ConvolutionalCodeDecoder convDecoder(n, k, m,ConvolutionalMatrix);

    // ���岻ͬ�� epsilon ֵ��������ʣ�
    vector<double> epsilons = {0.0, 0.01, 0.02, 0.03, 0.04, 0.05}; 
    vector<double> berLBC, berCC, berNoCode;

    // ģ�ⲻͬ epsilon �µ�����
    for (double epsilon : epsilons) {
        double ber1 = 0.0, ber2 = 0.0, ber3 = 0.0;

        // ���Է�����
        simulatePerformance(epsilon, lbcEncoder, lbcDecoder, infoBits, gen, ber1);
        berLBC.push_back(ber1);

        // �����
        simulatePerformance(epsilon, convEncoder, convDecoder, infoBits, gen, ber2);
        berCC.push_back(ber2);

        // �ޱ���
        vector<int> received = simulate_bsc(infoBits, epsilon);
        ber3 = calculateBER(infoBits, received);
        berNoCode.push_back(ber3);

        cout << "Epsilon: " << epsilon 
             << ", LBC BER: " << ber1 
             << ", CC BER: " << ber2 
             << ", No Code BER: " << ber3 << endl;
    }

    // �������� CSV �ļ�
    ofstream outFile("BER_comparison.csv");
    if (outFile.is_open()) {
        outFile << "Epsilon,LBC_BER,CC_BER,NoCode_BER\n";
        for (size_t i = 0; i < epsilons.size(); ++i) {
            outFile << epsilons[i] << "," << berLBC[i] << "," << berCC[i] << "," << berNoCode[i] << "\n";
        }
        outFile.close();
        cout << "Results saved to BER_comparison.csv" << endl;
    } else {
        cerr << "Unable to open file for writing." << endl;
    }

    return 0;
}