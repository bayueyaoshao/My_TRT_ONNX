#include "QtWidgets_RIAD.h"
#include<QDebug>
#include<QFileDialog>
#include <QElapsedTimer>

QtWidgets_RIAD::QtWidgets_RIAD(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    connect(ui.btn_inference, &QPushButton::clicked, this, &QtWidgets_RIAD::on_btn_inference);
    connect(ui.btn_openfile, &QPushButton::clicked, this, &QtWidgets_RIAD::on_btn_openfile);
    m_file_name = "D:/bottle_contamination_07.png";
    ui.lineEdit->setText(m_file_name.c_str());
    m_image_src = cv::imread(m_file_name.c_str());
    matToQimageLabelShow(ui.lbl_ori, m_image_src);


    m_ngCount = 0;
    m_maxValue = 0;
    m_minValue = 10;
}

QtWidgets_RIAD::~QtWidgets_RIAD()
{}

void QtWidgets_RIAD::matToQimageLabelShow(QLabel* label, cv::Mat& mat)
{
    cv::Mat Rgb;
    QImage Img;
    if (mat.channels() == 3)//RGB Img
    {
         cv::cvtColor(mat, Rgb, cv::COLOR_BGR2RGB);//颜色空间转换
         Img = QImage((const uchar*)(Rgb.data), Rgb.cols, Rgb.rows, Rgb.cols * Rgb.channels(), QImage::Format_RGB888);
    }
    else//Gray Img
    {
        Img = QImage((const uchar*)(mat.data), mat.cols, mat.rows, mat.cols * mat.channels(), QImage::Format_Indexed8);
    }
    label->setPixmap(QPixmap::fromImage(Img));
    label->setScaledContents(true);
}


void QtWidgets_RIAD::on_btn_openfile()
{
    QString file_name = QFileDialog::getOpenFileName(this,
        tr("Open File"),
        "",
        "",
        0);
    if (!file_name.isNull())
    {
        //fileName是文件名
        qDebug() << file_name;
        m_file_name = file_name.toStdString();
        ui.lineEdit->setText(file_name);
        m_image_src = cv::imread(m_file_name.c_str());
        matToQimageLabelShow(ui.lbl_ori, m_image_src);
    }   
}



// 单通道640图像
void QtWidgets_RIAD::on_btn_inference()
{
    bool isEngine = true;
    int image_size = 640;

    cv::Mat recMat;
    std::string file_name = "D:/Work/TensorRT-8.5.1.7/TensorRT-8.5.1.7/bin/419_HB_0815170556.png";
    if (isEngine)
    {
        std::string engine_model = "D:/Work/TensorRT-8.5.1.7/TensorRT-8.5.1.7/bin/419-HB-V167_6401.engine";
        loadEngineModel(engine_model, image_size);
    }
    else
    {
        std::string engine_model = "D:/Work/TensorRT-8.5.1.7/TensorRT-8.5.1.7/bin/419-HB-V167_6401.onnx";
        //std::string engine_model = "D:/Work/Code/tensorflow-unet/models/1/model.onnx";
        load_model(engine_model);
    }
    for (int i = 0; i < 100; i++)
    {
        QElapsedTimer timer; // 创建定时器
        timer.start(); // 开始计时
        
        cv::Mat result = model_inference(file_name, image_size, recMat, isEngine);
        qint64 elapsedTime = timer.nsecsElapsed(); // 获取纳秒为单位的持续时间
        //qDebug() << "Execution time: " << elapsedTime / 1e6 << " ms."; // 转换算成毫秒并输出
        ui.label_time->setText(QString("%1").arg(elapsedTime / 1e6));
        ui.textEdit->append(QString("%1").arg(elapsedTime / 1e6));
    }

}














