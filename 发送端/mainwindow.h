#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <boost/asio.hpp>
#include <memory>
#include <QLabel>
#include <QSpinBox>
#include <QTextEdit>
#include <QCheckBox>

using boost::asio::ip::tcp;

namespace Ui {
class MainWindow;
}
class Encoder;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void chooseFile();  // 添加这一行来声明 chooseFile 函数
    void sendData();    // 确保 sendData 函数也已声明

private slots:


private:
    QSpinBox *nSpinBox;
    QSpinBox *kSpinBox;
    QSpinBox *mSpinBox;
    QTextEdit *originalDataTextEdit;
    QTextEdit *encodedDataTextEdit;
    QDoubleSpinBox *errorProbabilitySpinBox;
    QComboBox *encodingMethodComboBox;
    QComboBox *inputTypeComboBox;
    QLineEdit *filePathLineEdit;
    QPushButton *chooseFileButton;
    QPushButton *sendButton;
    QTextEdit *statusTextEdit;
    QCheckBox *simulateBSCCheckBox;
};

#endif // MAINWINDOW_H
