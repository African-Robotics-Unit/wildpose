/********************************************************************************
** Form generated from reading UI file 'DialogOpenServer.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOGOPENSERVER_H
#define UI_DIALOGOPENSERVER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_DialogOpenServer
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *le_address;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *DialogOpenServer)
    {
        if (DialogOpenServer->objectName().isEmpty())
            DialogOpenServer->setObjectName(QStringLiteral("DialogOpenServer"));
        DialogOpenServer->resize(400, 89);
        verticalLayout = new QVBoxLayout(DialogOpenServer);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(DialogOpenServer);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        le_address = new QLineEdit(DialogOpenServer);
        le_address->setObjectName(QStringLiteral("le_address"));

        horizontalLayout->addWidget(le_address);


        verticalLayout->addLayout(horizontalLayout);

        buttonBox = new QDialogButtonBox(DialogOpenServer);
        buttonBox->setObjectName(QStringLiteral("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(DialogOpenServer);
        QObject::connect(buttonBox, SIGNAL(accepted()), DialogOpenServer, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), DialogOpenServer, SLOT(reject()));

        QMetaObject::connectSlotsByName(DialogOpenServer);
    } // setupUi

    void retranslateUi(QDialog *DialogOpenServer)
    {
        DialogOpenServer->setWindowTitle(QApplication::translate("DialogOpenServer", "Open server", Q_NULLPTR));
        label->setText(QApplication::translate("DialogOpenServer", "Address", Q_NULLPTR));
        le_address->setInputMask(QString());
        le_address->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class DialogOpenServer: public Ui_DialogOpenServer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOGOPENSERVER_H
