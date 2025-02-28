#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <boost/asio.hpp>
#include <memory>
#include <vector>

#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QLabel>
#include <QSpinBox>


namespace Ui {
class MainWindow;
}

class Dncoder;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    static std::string join(const std::vector<int>& vec);

private slots:
    void startReceiving();


private:
    QTextEdit *receivedDataTextEdit;
    QTextEdit *decodedDataTextEdit;
    QTextEdit *statusTextEdit;
    QComboBox *decodingMethodComboBox;
    QSpinBox *nSpinBox;
    QSpinBox *kSpinBox;
    QSpinBox *mSpinBox;
    QPushButton *startButton;

    void receiveData();
    QTextEdit *outputTextEdit;

};

#endif // MAINWINDOW_H
