#include "QtWidgets_RIAD.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QtWidgets_RIAD w;
    w.show();
    return a.exec();
}
