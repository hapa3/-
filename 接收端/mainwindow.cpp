#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QInputDialog>
#include <boost/asio.hpp>
#include <memory>
#include <vector>
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
#include <boost/asio/ip/tcp.hpp>
#include <exception>
#include <QFileDialog>
using namespace std;
using boost::asio::ip::tcp;
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
            bool flag = 0;  // 每次计算 syndrome 时重置 flag
            for (int j = 0; j < n; ++j) {
                flag ^= received[j] * HT[j][i];  // 使用按位异或运算符
            }
            syndrome[i] = (int)flag;
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

void sendErrorMessage(boost::asio::ip::tcp::socket& socket, const std::string& message) {
    boost::asio::write(socket, boost::asio::buffer(message));
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


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {

    // 创建控件
    receivedDataTextEdit = new QTextEdit(this);
    receivedDataTextEdit->setReadOnly(true);

    decodedDataTextEdit = new QTextEdit(this);
    decodedDataTextEdit->setReadOnly(true);

    statusTextEdit = new QTextEdit(this);
    statusTextEdit->setReadOnly(true);

    decodingMethodComboBox = new QComboBox(this);
    decodingMethodComboBox->addItem("线性分组码");
    decodingMethodComboBox->addItem("卷积码");

    nSpinBox = new QSpinBox(this);
    nSpinBox->setRange(1, 100);
    nSpinBox->setPrefix("n: ");

    kSpinBox = new QSpinBox(this);
    kSpinBox->setRange(1, 100);
    kSpinBox->setPrefix("k: ");

    mSpinBox = new QSpinBox(this);
    mSpinBox->setRange(1, 10);
    mSpinBox->setPrefix("m: ");
    mSpinBox->setEnabled(false); // 默认为禁用状态




    // 设置卷积码时启用 m 参数
    connect(decodingMethodComboBox, &QComboBox::currentIndexChanged, this, [this]() {
        mSpinBox->setEnabled(decodingMethodComboBox->currentText() == "卷积码");
    });

    startButton = new QPushButton("开始接收", this);
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startReceiving);

    // 设置布局
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(new QLabel("选择解码方法:"), 0, 0);
    layout->addWidget(decodingMethodComboBox, 0, 1);
    layout->addWidget(nSpinBox, 1, 0);
    layout->addWidget(kSpinBox, 1, 1);
    layout->addWidget(mSpinBox, 1, 2);

    layout->addWidget(new QLabel("接收到的数据:"), 2, 0);
    layout->addWidget(receivedDataTextEdit, 3, 0, 1, 3);
    layout->addWidget(new QLabel("解码后的数据:"), 4, 0);
    layout->addWidget(decodedDataTextEdit, 5, 0, 1, 3);
    layout->addWidget(new QLabel("状态信息:"), 6, 0);
    layout->addWidget(statusTextEdit, 7, 0, 1, 3);
    layout->addWidget(startButton, 8, 0, 1, 3);

    // 创建中央小部件并设置布局
    QWidget *centralWidget = new QWidget(this);
    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);

    // 设置窗口标题和大小
    setWindowTitle("接收端界面");
    setFixedSize(600, 500);
}

// 析构函数
MainWindow::~MainWindow() {}

// 开始接收数据
void MainWindow::startReceiving() {
    std::thread([this]() {

        receiveData(); // 持续接收数据

    }).detach(); // 在新线程中接收数据
}

std::string MainWindow::join(const std::vector<int>& vec) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << " "; // 用空格分隔
        }
    }
    return oss.str();
}

// 接收数据的方法
void MainWindow::receiveData() {
    boost::asio::io_context io_context;
    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 12345));
    tcp::socket socket(io_context);
    try {
        statusTextEdit->append("等待连接...");
        acceptor.accept(socket); // 阻塞，直到有连接
        statusTextEdit->append("连接成功！");

        // 获取解码参数
        int n = nSpinBox->value();
        int k = kSpinBox->value();
        int m = decodingMethodComboBox->currentText() == "卷积码" ? mSpinBox->value() : 0;

        // 创建解码器
        std::unique_ptr<Decoder> decoder;
        if (decodingMethodComboBox->currentText() == "线性分组码") {
            auto generatorMatrix = generateGeneratorMatrix(k, n, 13);
            decoder = std::make_unique<LinearBlockCodeDecoder>(k, n, generatorMatrix);
        } else {
            auto generatorMatrix = generateConvolutionalMatrix(n, k, m, 13);
            decoder = std::make_unique<ConvolutionalCodeDecoder>(n, k, m, generatorMatrix);
        }

        std::vector<int> received;
        int bit;
        std::vector<int> right;
        std::vector<int> decode_text;
        while (boost::asio::read(socket, boost::asio::buffer(&bit, sizeof(bit)))) {
            if (bit == 3) {
                break;
            }

            right.push_back(bit);
        }
        size_t data_length;
        boost::asio::read(socket, boost::asio::buffer(&data_length, sizeof(data_length)));
        std::vector<int> decoded_data;
        statusTextEdit->append("等待数据...");

        // 接收编码后的码字
        vector<int> received_data;
        while (true) {
            // 接收编码后的码字
            received_data.clear();
            while (boost::asio::read(socket, boost::asio::buffer(&bit, sizeof(bit)))) {
                if (bit >= 2) {
                    break;
                }

                received_data.push_back(bit);
            }
            decoded_data = decoder->decode(received_data);
            if (decoded_data.empty()) {
                // 检测到错误
                char ack = 'E';
                boost::asio::write(socket, boost::asio::buffer(&ack, sizeof(ack)));
            }
            if (decoded_data.size() > data_length) {
                decoded_data.resize(data_length);
            }
            if(calculateBER(right,decoded_data) > 0){
                char ack = 'E';
                boost::asio::write(socket, boost::asio::buffer(&ack, sizeof(ack)));
            }
            else{
                char ack = 'A';
                boost::asio::write(socket, boost::asio::buffer(&ack, sizeof(ack)));
                break;
            }
        }

        // 更新 UI
        QMetaObject::invokeMethod(this, [this, received_data = std::move(received_data), decoded_data = std::move(decoded_data)]() {
            receivedDataTextEdit->append("接收到的数据: " + QString::fromStdString(join(received_data)));
            decodedDataTextEdit->append("解码后的数据: " + QString::fromStdString(join(decoded_data)));
            statusTextEdit->append("数据接收成功！");

            // 选择保存数据
            QStringList options;
            options << "保存为文本" << "保存为文件" << "保存为图像";
            bool ok;
            QString selectedOption = QInputDialog::getItem(this, "选择保存方式", "保存方式:", options, 0, false, &ok);
            if (ok && !selectedOption.isEmpty()) {
                if (selectedOption == "保存为文本") {
                    std::string text = binary_to_string(decoded_data);
                    QString fileName = QFileDialog::getSaveFileName(this, "保存文本", "", "Text Files (*.txt)");
                    if (!fileName.isEmpty()) {
                        std::ofstream outFile(fileName.toStdString());
                        std::string text = binary_to_string(decoded_data);
                        outFile << text;
                        outFile.close();
                        statusTextEdit->append("文本已保存到: " + fileName);
                    }
                }
                else if (selectedOption == "保存为文件") {
                    QString fileName = QFileDialog::getSaveFileName(this, "保存文件", "", "All Files (*.*)");
                    if (!fileName.isEmpty()) {
                        binary_to_file(decoded_data, fileName.toStdString());
                        statusTextEdit->append("数据已保存到: " + fileName);
                    }
                }
                else if (selectedOption == "保存为图像") {
                    QString fileName = QFileDialog::getSaveFileName(this, "保存图像", "", "Image Files (*.png *.jpg *.bmp)");
                    if (!fileName.isEmpty()) {
                        binary_to_file(decoded_data, fileName.toStdString());
                        statusTextEdit->append("图像已保存到: " + fileName);
                    }
                }
            }
        });
    }
    catch (std::exception &e) {
        QMetaObject::invokeMethod(this, [this, e]() {
            statusTextEdit->append("错误: " + QString::fromStdString(e.what()));
        });
    }
}






