#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtWidgets_RIAD.h"
#include"tensorrt_api/tensor_base.h"
#include"tensorrt_api/tensor_yolo5.h"
#include"tensorrt_api/tensor_RIAD.h"
#include "riad_alg.h"
class QtWidgets_RIAD : public QMainWindow
{
    Q_OBJECT

public:
    QtWidgets_RIAD(QWidget *parent = nullptr);
    ~QtWidgets_RIAD();

private:
    Ui::QtWidgets_RIADClass ui;
    std::string m_file_name;
    cv::Mat m_image_src;
    void matToQimageLabelShow(QLabel* label, cv::Mat& mat);
    int m_ngCount;
    double m_maxValue;
    double m_minValue;


private slots:
    void on_btn_inference();
    void on_btn_openfile();





};
