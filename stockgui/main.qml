import QtQuick 2.9
import QtQuick.Controls 2.2

Rectangle {
    id: root
    width: 1024; height: 768
    color: "lightgray"

    Image {
        id: image1
        x: 0
        y: 0
        width: 1024
        height: 768
        clip: false
        visible: true
        fillMode: Image.Stretch
        z: 0
        sourceSize.width: 791
        source: "red.png"

        Image {
            id: predictpic
            x: 667
            y: 26
            width: 340
            height: 340
            visible: true
            source: "qrc:/qtquickplugin/images/template_image.png"
        }

        Image {
            id: stockpic
            x: 466
            y: 153
            width: 72
            height: 72
            z: 2
            sourceSize.height: 1024
            sourceSize.width: 1024
            scale: 4
            source: stock.return_picaddr()
        }

        Text {
            id: today
            x: 177
            y: 21
            width: 161
            height: 41
            text: stock.return_today()
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            fontSizeMode: Text.HorizontalFit
            z: 5
            style: Text.Normal
            font.pixelSize: 25
        }

        Button {
            id: picdown
            x: 546
            y: 364
            text: qsTr("up")
            font.family: "Tahoma"
            clip: false
            visible: true
            font.pointSize: 18
            highlighted: false
        }

        Button {
            id: picup
            x: 363
            y: 364
            text: qsTr("down")
            font.family: "Tahoma"
            highlighted: false
            font.pointSize: 18
        }


        Slider {
            id: slider
            x: 358
            y: 410
            width: 288
            height: 63
            stepSize: 1
            to: stock.return_maxpicnum()
            value: 0
            onValueChanged: {
                stock.set_today(value)
                stockpic.source =  stock.return_picaddr()
                today.text = stock.return_today()
                todayprice.text = stock.return_todayprice()
                income.text = stock.return_income()
            }

        }

        BusyIndicator {
            id: busyIndicator
            x: 678
            y: 410
            width: 296
            height: 323
            opacity: 0.5
            running: false
            onRunningChanged: {

            }

            Button {
                id: downloaddata
                x: 22
                y: 16
                width: 247
                height: 51
                text: qsTr("Button")
                checkable: true
                opacity: 0
                visible: true
                onClicked: {
                    busyIndicator.running = true
                    stock.downloaddata()
                }

            }

            Button {
                id: processdata
                x: 22
                y: 78
                width: 247
                height: 51
                text: qsTr("Button")
                opacity: 0
                visible: true
                onClicked: {
                    busyIndicator.running = true
                    stock.processdata()

                }
            }


            Button {
                id: tablelize
                x: 22
                y: 135
                width: 247
                height: 56
                text: qsTr("Button")
                opacity: 0
                visible: true
                onClicked: {
                    busyIndicator.running = true
                    stock.tablelize()
                }
            }


            Button {
                id: drawpic
                x: 22
                y: 198
                width: 247
                height: 51
                text: qsTr("Button")
                opacity: 0
                visible: true
                onClicked: {
                    busyIndicator.running = true
                    stock.drawpic()
                }
            }


            Button {
                id: predict
                x: 22
                y: 259
                width: 247
                height: 51
                text: ""
                checked: true
                highlighted: true
                opacity: 0
                visible: true
                onClicked: {
                    busyIndicator.running = true
                    stock.predict()
                }
            }


        }


        Switch {
            id: switch1
            x: 546
            y: 494
            width: 110
            height: 44
            text: qsTr("")
            checked: false
            onPositionChanged: {
                changetextarea.visible=switch1.checked
                scrollView.visible=switch1.checked
                textArea.visible=switch1.checked
                textArea.text=stock.showstocklist()
            }

        }

        Image {
            id: changetextarea
            x: 678
            y: 408
            width: 296
            height: 325
            visible: false
            source: "changetextarea.png"

            ScrollView {
                id: scrollView
                x: 0
                y: 0
                width: 296
                height: 325
                visible: false


                TextArea {
                    id: textArea
                    x: 0
                    y: 0
                    width: 296
                    height: 325
                    text: qsTr("Text Area\n")
                    visible: false
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.pointSize: 18
                }
            }
        }

        Button {
            id: buy
            x: 378
            y: 499
            width: 100
            height: 57
            text: qsTr("Button")
            opacity: 0
            visible: true
            onClicked: {
                stock.buystock(numinput.text)
                textArea.text=stock.showstocklist()
                money.text= stock.return_money()
                income.text = stock.return_income()
            }
        }

        Button {
            id: sell
            x: 378
            y: 564
            width: 100
            height: 59
            text: qsTr("Button")
            opacity: 0
            visible: true
            onClicked: {
                stock.sellstock(valueinput.text,numinput.text)
                textArea.text=stock.showstocklist()
                money.text= stock.return_money()
                income.text = stock.return_income()
            }
        }

        TextInput {
            id: numinput
            x: 375
            y: 685
            width: 122
            height: 48
            text: qsTr("")
            font.family: "Tahoma"
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: 30
        }

        TextInput {
            id: valueinput
            x: 511
            y: 685
            width: 122
            height: 48
            text: qsTr("")
            horizontalAlignment: Text.AlignHCenter
            font.family: "Tahoma"
            font.pixelSize: 30
        }

        Text {
            id: money
            x: 177
            y: 69
            width: 161
            height: 41
            text: stock.return_money()
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            fontSizeMode: Text.HorizontalFit
            font.pixelSize: 25
        }

        Text {
            id: todayprice
            x: 177
            y: 116
            width: 161
            height: 41
            text: stock.return_todayprice()
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            fontSizeMode: Text.HorizontalFit
            font.pixelSize: 25
        }

        Text {
            id: income
            x: 177
            y: 163
            width: 161
            height: 41
            text: stock.return_income()
            fontSizeMode: Text.HorizontalFit
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: 25
        }





    }

    Connections {
        target: picdown
        onClicked: {
            stock.add_today()
            stockpic.source =  stock.return_picaddr()
            today.text = stock.return_today()
            todayprice.text = stock.return_todayprice()
            income.text = stock.return_income()
        }
    }

    Connections {
        target: picup
        onClicked: {
            stock.minus_today()
            stockpic.source =  stock.return_picaddr()
            today.text = stock.return_today()
            todayprice.text = stock.return_todayprice()
            income.text = stock.return_income()
        }
    }

    Connections{
        target: stock

        onBusysig:{
            busyIndicator.running=indicator
            slider.to=stock.return_maxpicnum()
        }
        onPicsig:{
            predictpic.source=predict_pic
        }
    }



}
