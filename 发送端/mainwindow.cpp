#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <boost/asio.hpp>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <iomanip>
#include <QInputDialog>
using namespace std;
using boost::asio::ip::tcp;
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
    std::uniform_real_distribution<> dis(0.0, 1.0);

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

MainWindow::~MainWindow() {}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {
    // 创建控件
    encodingMethodComboBox = new QComboBox(this);
    encodingMethodComboBox->addItem("线性分组码");
    encodingMethodComboBox->addItem("卷积码");

    inputTypeComboBox = new QComboBox(this);
    inputTypeComboBox->addItem("文本");
    inputTypeComboBox->addItem("文件");
    inputTypeComboBox->addItem("图像");

    filePathLineEdit = new QLineEdit(this);
    chooseFileButton = new QPushButton("选择文件", this);
    connect(chooseFileButton, &QPushButton::clicked, this, &MainWindow::chooseFile);

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
    connect(encodingMethodComboBox, &QComboBox::currentIndexChanged, this, [this]() {
        mSpinBox->setEnabled(encodingMethodComboBox->currentText() == "卷积码");
    });

    simulateBSCCheckBox = new QCheckBox("模拟 BSC", this);
    errorProbabilitySpinBox = new QDoubleSpinBox(this);
    errorProbabilitySpinBox->setRange(0.0, 0.5);
    errorProbabilitySpinBox->setPrefix("错误概率: ");
    errorProbabilitySpinBox->setSingleStep(0.01);
    errorProbabilitySpinBox->setValue(0.1);
    errorProbabilitySpinBox->setEnabled(false); // 默认禁用

    connect(simulateBSCCheckBox, &QCheckBox::toggled, errorProbabilitySpinBox, &QDoubleSpinBox::setEnabled);



    sendButton = new QPushButton("启动发送", this);

    originalDataTextEdit = new QTextEdit(this);
    encodedDataTextEdit = new QTextEdit(this);
    statusTextEdit = new QTextEdit(this);

    // 设置布局
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(new QLabel("选择编码方法:"), 0, 0);
    layout->addWidget(encodingMethodComboBox, 0, 1);
    layout->addWidget(nSpinBox, 1, 0);
    layout->addWidget(kSpinBox, 1, 1);
    layout->addWidget(mSpinBox, 1, 2);

    layout->addWidget(new QLabel("选择输入类型:"), 2, 0);
    layout->addWidget(inputTypeComboBox, 2, 1);
    layout->addWidget(filePathLineEdit, 3, 0, 1, 2);
    layout->addWidget(chooseFileButton, 3, 2);

    layout->addWidget(simulateBSCCheckBox, 4, 0);
    layout->addWidget(errorProbabilitySpinBox, 4, 1);

    layout->addWidget(new QLabel("原始数据:"), 5, 0);
    layout->addWidget(originalDataTextEdit, 6, 0, 1, 3);
    layout->addWidget(new QLabel("编码数据:"), 7, 0);
    layout->addWidget(encodedDataTextEdit, 8, 0, 1, 3);
    layout->addWidget(new QLabel("状态信息:"), 9, 0);
    layout->addWidget(statusTextEdit, 10, 0, 1, 3);

    layout->addWidget(sendButton, 11, 0, 1, 3);

    // 创建中央小部件并设置布局
    QWidget *centralWidget = new QWidget(this);
    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);

    // 设置窗口标题和大小
    setWindowTitle("发送端界面");
    setFixedSize(600, 500);

    // 连接信号和槽
    connect(sendButton, &QPushButton::clicked, this, &MainWindow::sendData);
}

std::string join(const std::vector<int>& values, const std::string& delimiter = ", ") {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        oss << values[i];
        if (i != values.size() - 1) {
            oss << " "; // 添加分隔符
        }
    }
    return oss.str();
}

// 选择文件
void MainWindow::chooseFile() {
    QString filename = QFileDialog::getOpenFileName(this, "选择文件");
    if (!filename.isEmpty()) {
        filePathLineEdit->setText(filename);
    }
}

// 发送数据
void MainWindow::sendData() {
    try {
        // 创建 IO 上下文和 socket 连接
        boost::asio::io_context io_context;
        tcp::resolver resolver(io_context);
        tcp::resolver::results_type endpoints = resolver.resolve("127.0.0.1", "12345");
        tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints);

        // 选择编码方法
        std::unique_ptr<Encoder> encoder;
        int n = nSpinBox->value();
        int k = kSpinBox->value();
        int m = mSpinBox->value();
        QString encodingMethod = encodingMethodComboBox->currentText();

        if (encodingMethod == "线性分组码") {
            auto generatorMatrix = generateGeneratorMatrix(k, n, 13);
            encoder = std::make_unique<LinearBlockCodeEncoder>(k, n, generatorMatrix);
        } else if (encodingMethod == "卷积码") {
            auto generatorMatrix = generateConvolutionalMatrix(n, k, m, 13);
            encoder = std::make_unique<ConvolutionalCodeEncoder>(n, k, m, generatorMatrix);
        } else {
            statusTextEdit->append("无效的编码方法选择！");
            return;
        }

        // 获取输入类型
        QString inputType = inputTypeComboBox->currentText();
        std::vector<int> binary_data;

        if (inputType == "文本") {
            QString text = QInputDialog::getText(this, "输入文本", "请输入文本:");
            binary_data = string_to_binary(text.toStdString());
        } else if (inputType == "文件" || inputType == "图像") {
            QString filePath = filePathLineEdit->text();
            if (filePath.isEmpty()) {
                QMessageBox::warning(this, "警告", "请选择文件！");
                return;
            }

            // 检查文件是否存在
            QFile file(filePath);
            if (!file.exists()) {
                QMessageBox::warning(this, "警告", "文件不存在！");
                return;
            }

            // 直接读取文件内容并转换为二进制数据
            std::ifstream inputFile(filePath.toStdString(), std::ios::binary);
            if (!inputFile) {
                QMessageBox::warning(this, "警告", "无法打开文件！");
                return;
            }

            char byte;
            while (inputFile.get(byte)) {
                for (int i = 7; i >= 0; --i) {
                    binary_data.push_back((byte >> i) & 1);
                }
            }

            if (binary_data.empty()) {
                QMessageBox::warning(this, "警告", "文件内容无效！");
                return;
            }
        } else {
            statusTextEdit->append("无效的输入类型选择！");
            return;
        }

        // 显示原始数据
        originalDataTextEdit->setPlainText(QString::fromStdString(join(binary_data)));

        // 检查数据块大小并补齐
        if (binary_data.size() % encoder->get_block_size() != 0) {
            for (int i = 0; i < encoder->get_block_size() - binary_data.size() % encoder->get_block_size(); i++) {
                binary_data.push_back(0);
            }
        }

        // 编码数据
        std::vector<int> encoded_data = encoder->encode(binary_data);
        encodedDataTextEdit->setPlainText(QString::fromStdString(join(encoded_data)));

        for (int bit : binary_data) {
            boost::asio::write(socket, boost::asio::buffer(&bit, sizeof(bit)));
        }

        int end_of_data = 3 ;
        boost::asio::write(socket, boost::asio::buffer(&end_of_data, sizeof(end_of_data)));
        size_t data_length = binary_data.size();
        boost::asio::write(socket, boost::asio::buffer(&data_length, sizeof(data_length)));
        if(binary_data.size() % encoder->get_block_size() != 0){
            for(int i = 0; i < encoder->get_block_size() - binary_data.size() % encoder->get_block_size(); i++){
                binary_data.push_back(0);
            }
        }

        // 设置错误概率并模拟传输信道
        bool add_simulate_bsc = simulateBSCCheckBox->isChecked();
        std::vector<int> transmitted_codeword;

        // 设置错误概率并模拟传输信道
        if (add_simulate_bsc) {
            double error_probability = errorProbabilitySpinBox->value();
            transmitted_codeword = simulate_bsc(encoded_data, error_probability);
        } else {
            transmitted_codeword = encoded_data;
        }

        bool success = false;
        while (!success) {
            // 发送编码后的码字
            for (int bit : transmitted_codeword) {
                boost::asio::write(socket, boost::asio::buffer(&bit, sizeof(bit)));
            }

            // 发送数据传输完成标志
            end_of_data = 2;
            boost::asio::write(socket, boost::asio::buffer(&end_of_data, sizeof(end_of_data)));

            // 等待接收端的确认
            char ack;
            boost::asio::read(socket, boost::asio::buffer(&ack, sizeof(ack)));

            if (ack == 'A') {
                statusTextEdit->append("数据已成功接收！");
                success = true;
                break;
            }
            else if (ack == 'E') {
                statusTextEdit->append("接收端检测到错误，重新发送数据...");
                // 重新生成带有错误的传输码字
                if(add_simulate_bsc){
                    double error_probability = errorProbabilitySpinBox->value();
                    transmitted_codeword = simulate_bsc(encoded_data, error_probability);
                }
                else transmitted_codeword = encoded_data;
            }
            else {
                statusTextEdit->append("未知的确认信号！");
                break;
            }
        }
        end_of_data = 4;
        boost::asio::write(socket, boost::asio::buffer(&end_of_data, sizeof(end_of_data)));

    } catch (std::exception &e) {
        statusTextEdit->append(QString("错误: %1").arg(e.what()));
    }
}








