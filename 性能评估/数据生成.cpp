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

// 编码器接口
class Encoder {
public:
    virtual std::vector<int> encode(const std::vector<int>& message) = 0;
    virtual size_t get_block_size() const = 0;
    virtual ~Encoder() = default;
};

// 线性分组码编码器
class LinearBlockCodeEncoder : public Encoder {
public:

    LinearBlockCodeEncoder(int k, int n,vector<vector<int>> generatorMatrix) 
        : k(k), n(n),generatorMatrix(generatorMatrix) {
        if (!validateBlockCode(k, n)) {
            throw std::invalid_argument("n 必须大于 k");
        }
    }
    std::vector<int> encode(const std::vector<int>& input) override {
        vector<vector<int>> result;
        for (size_t i = 0; i < input.size(); i += k) {
            // 从原始向量中截取长度为 n 的子向量，注意最后一段可能不足 n
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

        // 线性分组码编码：将输入比特和生成矩阵相乘
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                output[j] ^= block[i] * generatorMatrix[i][j];  // 模2加法
            }
        }
        return output;
    }
    // 判断矩阵是否有k行n列
    bool validateBlockCode(int k, int n) {
        return (n > k);  // n 必须大于 k
    }

};

// 卷积码编码器
class ConvolutionalCodeEncoder : public Encoder {
private:
    int n; // 输出比特数
    int k; // 输入比特数
    int m; // 寄存器长度

    // 生成多项式矩阵，大小为 n x (m+1) x k
    vector<vector<vector<int>>> generatorMatrix;

    // 寄存器数组，用于存储 k-bit 输入序列
    vector<vector<int>> shiftRegister;

public:
    // 构造函数
    ConvolutionalCodeEncoder(int n, int k, int m, vector<vector<vector<int>>> generatorMatrix)
        : n(n), k(k), m(m), generatorMatrix(generatorMatrix) {
        shiftRegister = vector<vector<int>>(m + 1, vector<int>(k, 0));
    }

    // 更新寄存器的内容
    void shiftRegisters(const vector<int>& inputBits) {
        // 移动寄存器内容
        for (int i = m; i > 0; i--) {
            shiftRegister[i] = shiftRegister[i - 1];
        }
        // 将输入放入第一个寄存器
        for (int i = 0; i < k; i++) {
            shiftRegister[0][i] = inputBits[i];
        }
    }

    // 根据生成多项式计算输出
    vector<int> encode2(const vector<int>& inputBits){
        vector<int> outputBits(n, 0);

        // 更新寄存器内容
        shiftRegisters(inputBits);

        // 计算输出比特
        for (int i = 0; i < n; i++) {          // 对于每个输出比特
            for (int j = 0; j <= m; j++) {      // 对于每个寄存器
                for (int l = 0; l < k; l++) {   // 对于寄存器中每个比特
                    outputBits[i] ^= shiftRegister[j][l] & generatorMatrix[i][j][l];
                }
            }
        }

        return outputBits;
    }

    // 编码整个输入序列，并加入尾比特终止
        vector<int> encode(const vector<int>& inputBitsSequence) override {
        vector<int> encodedSequence;
        std::vector<std::vector<int>> result;
        // 遍历整个一维向量
        for (size_t i = 0; i < inputBitsSequence.size(); i += k) {
            // 从原始向量中截取长度为 n 的子向量，注意最后一段可能不足 n
            std::vector<int> segment(inputBitsSequence.begin() + i, inputBitsSequence.begin() + std::min(i + k, inputBitsSequence.size()));
            result.push_back(segment);
        }
        for (const auto& inputBits : result) {
            vector<int> outputBits = encode2(inputBits);
            encodedSequence.insert(encodedSequence.end(), outputBits.begin(), outputBits.end());
        }
        // // 加入尾比特终止
        // vector<int> zeroBits(k, 0);
        // for (int i = 0; i < m; i++) {
        //     encodedSequence.push_back(encode(zeroBits));
        // }
        return encodedSequence;
    }

    // 实现 get_block_size 函数
    size_t get_block_size() const override {
        return k; // 返回输入比特数作为块大小
    }
};

// 模拟二元对称信道
std::vector<int> simulate_bsc(const std::vector<int>& codeword, double error_probability) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::vector<int> transmitted_codeword = codeword;

    for (int& bit : transmitted_codeword) {
        if (dis(gen) < error_probability) {
            bit ^= 1; // 翻转位
        }
    }

    return transmitted_codeword;
}

// 将字符串转换为二进制向量
std::vector<int> string_to_binary(const std::string& str) {
    std::vector<int> binary;
    for (char c : str) {
        for (int i = 7; i >= 0; --i) {
            binary.push_back((c >> i) & 1);
        }
    }
    return binary;
}

// 将文件内容转换为二进制向量
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


// 自定义哈希函数
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

// 自定义相等比较函数
struct VectorEqual {
    bool operator()(const vector<int>& lhs, const vector<int>& rhs) const {
        return lhs == rhs;
    }
};


// 解码器接口
class Decoder {
public:
    virtual std::vector<int> decode(const std::vector<int>& received) = 0;
    virtual ~Decoder() = default;
};

// 线性分组码解码器 //辛德拉姆译码
class LinearBlockCodeDecoder : public Decoder {
public:
    LinearBlockCodeDecoder(int k, int n, const vector<vector<int>>& generatorMatrix) 
        : k(k), n(n), generatorMatrix(generatorMatrix) {
        if (!validateBlockCode(k, n)) {
            throw std::invalid_argument("n 必须大于 k");
        }
    }
    vector<int> decode(const vector<int>& receivedBits) override
     {
        vector<vector<int>> H = generateCheckMatrix(generatorMatrix, k, n);
        unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual> syndromeTable = generateSyndromeTable(H, n);
        std::vector<std::vector<int>> received;
    
        // 遍历整个一维向量
        for (size_t i = 0; i < receivedBits.size(); i += n) {
            // 从原始向量中截取长度为 n 的子向量，注意最后一段可能不足 n
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

    // 辛德拉姆译码
    vector<int> Sdecode(const vector<int>& received, const vector<vector<int>>& H, const unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual>& syndromeTable) {
        
        // 计算校验子
        vector<int> syndrome = calculateSyndrome(received, H);

        // 查找错误模式
        vector<int> errorPattern = findErrorPattern(syndrome, syndromeTable);

        // 如果没有找到错误模式，认为没有错误
        if (errorPattern.empty()) {
            return received;
        }

        // 修正错误：接收码字与错误模式进行模2加法
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

    // 判断矩阵是否有k行n列
    bool validateBlockCode(int k, int n) {
        return (n > k);  // n 必须大于 k
    }

    // 生成校验矩阵 H = (P^T | I)
    vector<vector<int>> generateCheckMatrix(const vector<vector<int>>& G, int k, int n) {
        int rows = n - k; // H 的行数
        int cols = n;     // H 的列数
        vector<vector<int>> H(rows, vector<int>(cols, 0));

        // 提取生成矩阵 G 中的右侧部分 P
        vector<vector<int>> P(k, vector<int>(rows, 0));
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < rows; ++j) {
                P[i][j] = G[i][k + j];  // 提取 G 的右侧部分 P
            }
        }

        // 转置 P 矩阵并填入 H 的左侧
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < k; ++j) {
                H[i][j] = P[j][i];  // P 的转置
            }
        }

        // 生成校验矩阵的右侧单位矩阵 I
        for (int i = 0; i < rows; ++i) {
            H[i][k + i] = 1;  // 在 H 的右侧添加单位矩阵部分
        }

        return H;
    }


    // 计算校验矩阵的转置
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

    // 计算校验子 S = r * H^T
    vector<int> calculateSyndrome(const vector<int>& received, const vector<vector<int>>& H) {
        int n = H[0].size();  // H 的列数（即码字的长度）
        int m = H.size();     // H 的行数（校验方程的数量）

        // 获取 H 的转置矩阵 H^T
        vector<vector<int>> HT = transposeMatrix(H);

        // 计算校验子 S = r * H^T
        vector<int> syndrome(m, 0);  // 初始化校验子

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                syndrome[i] ^= received[j] * HT[i][j];  // 模2加法
            }
        }
        
        return syndrome;
    }


    // 查找错误模式，根据校验子返回对应的错误模式
    vector<int> findErrorPattern(const vector<int>& syndrome, const unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual>& syndromeTable) {
        auto it = syndromeTable.find(syndrome);
        if (it != syndromeTable.end()) {
            return it->second;  // 找到对应的错误模式
        }
        return vector<int>();  // 没找到错误模式，返回空
    }

    // 生成辛德拉姆查找表
    unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual> generateSyndromeTable(const vector<vector<int>>& H, int n) {
        unordered_map<vector<int>, vector<int>, VectorHash, VectorEqual> syndromeTable;

        // 遍历所有可能的单比特错误模式
        for (int i = 0; i < n; ++i) {
            // 创建一个错误模式，只有第i位为1，表示第i位发生错误
            vector<int> errorPattern(n, 0);
            errorPattern[i] = 1;

            // 计算该错误模式对应的校验子
            vector<int> syndrome = calculateSyndrome(errorPattern, H);

            // 将校验子与错误模式映射到表中
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

    // 状态转移和输出
    vector<vector<int>> stateTransition;
    vector<vector<vector<int>>> outputBits;

    // 初始化状态转移和输出
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

    // 辅助函数：将十进制状态转换为二进制比特流矢量
    vector<int> decimalToBinary(int state, int bits) {
        vector<int> binary(bits, 0);
        for (int i = 0; i < bits; ++i) {
            binary[bits - 1 - i] = (state >> i) & 1;
        }
        return binary;
    }

    // 根据当前状态和输入计算输出
    vector<int> calculateOutput(int state, int input) {
        vector<int> output(n, 0);
        vector<int> reg(k, 0);
        for (int i = 0; i < n; ++i) {
            int outBit = 0;
            for (int j = 0; j < m; ++j) {
                reg = decimalToBinary((state >> ((m - j - 1) * k)) & ((1 << k) - 1), k); // 获取寄存器中的矢量状态
                for (int l = 0; l < k; ++l) {
                    outBit ^= reg[l] & generatorMatrix[i][j+1][l];
                }
            }
            // 对当前输入矢量进行处理
            reg = decimalToBinary(input, k);
            for (int l = 0; l < k; ++l) {
                outBit ^= reg[l] & generatorMatrix[i][0][l];
            }
            output[i] = outBit;
        }
        return output;
    }

    // 计算路径的汉明距离
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
        numStates = 1 << (k * m); // 更新状态数
        initStateTransition();
    }

    // Viterbi 译码过程
        vector<int> decode(const vector<int>& receivedBits) override {
        std::vector<std::vector<int>> received;
    
        // 遍历整个一维向量
        for (size_t i = 0; i < receivedBits.size(); i += n) {
            // 从原始向量中截取长度为 n 的子向量，注意最后一段可能不足 n
            std::vector<int> segment(receivedBits.begin() + i, receivedBits.begin() + std::min(i + n, receivedBits.size()));
            received.push_back(segment);
        }
        int T = received.size();
        vector<vector<int>> pathMetric(2, vector<int>(numStates, numeric_limits<int>::max()));
        vector<vector<int>> prevState(numStates, vector<int>(T, -1));

        pathMetric[0][0] = 0; // 初始状态

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

// 将二进制向量转换为字符串
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

// 将二进制向量写入文件
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


// 随机生成卷积码的生成矩阵G（n行m+1列，每个元素是一个k维矢量）
vector<vector<vector<int>>> generateConvolutionalMatrix(int n, int k, int m,unsigned int seed) {
    vector<vector<vector<int>>> G(n, vector<vector<int>>(m+1, vector<int>(k)));
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, 1);

    // 填充生成多项式矩阵G
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= m; ++j) {
            for (int l = 0; l < k; ++l) {
                G[i][j][l] = dis(gen);
            }
        }
    }

    return G;
}

// 检查矩阵行的线性无关性，通过增广矩阵的方法
bool isLinearlyIndependent(const vector<vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<int>> tempMatrix = matrix;

    for (int col = 0, row = 0; col < cols && row < rows; ++col) {
        int selectedRow = row;
        // 找到第一个该列中不为零的行
        for (int i = row; i < rows; ++i) {
            if (tempMatrix[i][col] == 1) {
                selectedRow = i;
                break;
            }
        }

        if (tempMatrix[selectedRow][col] == 0) {
            continue; // 如果整列为0，跳过该列
        }

        // 交换当前行和 selectedRow 行
        swap(tempMatrix[row], tempMatrix[selectedRow]);

        // 将下方所有行消去，确保每列的该行只有一个 1
        for (int i = row + 1; i < rows; ++i) {
            if (tempMatrix[i][col] == 1) {
                for (int j = col; j < cols; ++j) {
                    tempMatrix[i][j] ^= tempMatrix[row][j];
                }
            }
        }

        ++row; // 进入下一行
    }

    // 如果阶数等于行数，则线性无关
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

// 生成随机生成矩阵 (n, k) 线性分组码
vector<vector<int>> generateGeneratorMatrix(int k, int n, unsigned int seed) {
    // 初始化生成矩阵 G，大小为 k x n
    vector<vector<int>> G(k, vector<int>(n, 0));

    // 左侧部分为单位矩阵 I
    for (int i = 0; i < k; ++i) {
        G[i][i] = 1;
    }

    // 右侧部分为随机矩阵 P (大小为 k x (n - k))，并确保线性无关
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < k; ++i) {
        vector<int> tempRow(n - k, 0);
        do {
            for (int j = 0; j < n - k; ++j) {
                tempRow[j] = dis(gen); // 随机生成 0 或 1
            }

            // 复制临时行到矩阵的右侧
            for (int j = k; j < n; ++j) {
                G[i][j] = tempRow[j - k];
            }
        } while (!isLinearlyIndependent(G)); // 线性相关时，重新生成
    }

    return G;
}
// 二元对称信道仿真
vector<int> transmitThroughBSC(const vector<int>& codeword, double epsilon, mt19937& gen) {
    uniform_real_distribution<> dis(0, 1);
    vector<int> received = codeword;
    for (auto& bit : received) {
        if (dis(gen) < epsilon) {
            bit = 1 - bit; // 比特翻转
        }
    }
    return received;
}

// 计算误比特率
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
    // 参数设置
    int L = 1000; // 信息比特长度
    unsigned int seed = 42; // 随机种子
    mt19937 gen(seed);
    
    // 生成随机信息比特
    vector<int> infoBits(L);
    uniform_int_distribution<> dis(0, 1);
    for (int& bit : infoBits) {
        bit = dis(gen);
    }

    // 不编码的性能
    Encoder* noEncoder = nullptr;
    Decoder* noDecoder = nullptr;

    int n = 7, k = 4, m = 3;
    vector<vector<int>> GeneratorMatrix = generateGeneratorMatrix(k, n, 13);

    LinearBlockCodeEncoder lbcEncoder(k,n, GeneratorMatrix);
    LinearBlockCodeDecoder lbcDecoder(k,n,GeneratorMatrix);
    
    vector<vector<vector<int>>> ConvolutionalMatrix = generateConvolutionalMatrix(n, k, m,13);
    ConvolutionalCodeEncoder convEncoder(n,k,m,ConvolutionalMatrix);
    ConvolutionalCodeDecoder convDecoder(n, k, m,ConvolutionalMatrix);

    // 定义不同的 epsilon 值（错误概率）
    vector<double> epsilons = {0.0, 0.01, 0.02, 0.03, 0.04, 0.05}; 
    vector<double> berLBC, berCC, berNoCode;

    // 模拟不同 epsilon 下的性能
    for (double epsilon : epsilons) {
        double ber1 = 0.0, ber2 = 0.0, ber3 = 0.0;

        // 线性分组码
        simulatePerformance(epsilon, lbcEncoder, lbcDecoder, infoBits, gen, ber1);
        berLBC.push_back(ber1);

        // 卷积码
        simulatePerformance(epsilon, convEncoder, convDecoder, infoBits, gen, ber2);
        berCC.push_back(ber2);

        // 无编码
        vector<int> received = simulate_bsc(infoBits, epsilon);
        ber3 = calculateBER(infoBits, received);
        berNoCode.push_back(ber3);

        cout << "Epsilon: " << epsilon 
             << ", LBC BER: " << ber1 
             << ", CC BER: " << ber2 
             << ", No Code BER: " << ber3 << endl;
    }

    // 输出结果到 CSV 文件
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