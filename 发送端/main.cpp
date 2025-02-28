#include <QApplication>
#include <QTranslator>
#include <QLocale>
#include <QStringList>
#include "mainwindow.h" // 包含 MainWindow 的头文件

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // 设置全局样式
    a.setStyleSheet(R"(
        QWidget {
            background-color: #F0F4F8; /* 背景色，淡蓝色 */
            font-family: Arial; /* 字体 */
            font-size: 12px; /* 字号 */
        }
        QPushButton {
            border-radius: 5px;
            background-color: #A2C2E0; /* 按钮背景色，浅蓝色 */
            color: #FFFFFF; /* 按钮文字颜色，白色 */
            padding: 8px; /* 按钮内边距 */
            border: none; /* 去掉边框 */
        }
        QPushButton:hover {
            background-color: #8DA9B0; /* 鼠标悬停时的颜色 */
        }
        QTextEdit, QLineEdit, QSpinBox, QComboBox {
            border: 1px solid #C4D6E1; /* 控件边框，淡灰蓝色 */
            border-radius: 3px; /* 控件圆角 */
            padding: 6px; /* 内边距 */
            background-color: #FFFFFF; /* 控件背景色，白色 */
        }
        QLabel {
            font-weight: bold; /* 标签加粗 */
            color: #3A3A3A; /* 标签文字颜色，深灰色 */
        }
    )");

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "qtjieshouduan_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }

    MainWindow w;
    w.show();
    return a.exec();
}
