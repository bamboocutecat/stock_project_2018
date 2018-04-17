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
        source: "red.jpg"

        Image {
            id: predictpic
            x: 681
            y: 82
            width: 287
            height: 284
            visible: false
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
            x: 22
            y: 33
            width: 292
            height: 51
            text: stock.return_today()
            fontSizeMode: Text.HorizontalFit
            z: 5
            style: Text.Normal
            font.family: "Courier"
            font.pixelSize: 34
        }

        Button {
            id: picdown
            x: 546
            y: 364
            text: qsTr("up")
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
                    busyIndicator.running = stock.processdata()

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
                    busyIndicator.running = stock.tablelize()
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
                    busyIndicator.running = stock.drawpic()
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
                    busyIndicator.running = stock.predict()
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
            }

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

        Button {
            id: buy
            x: 380
            y: 505
            width: 100
            height: 57
            text: qsTr("Button")
            opacity: 0
            visible: true
        }

        Button {
            id: sell
            x: 380
            y: 582
            width: 100
            height: 59
            text: qsTr("Button")
            opacity: 0
            visible: true
        }





    }

    Connections {
        target: picdown
        onClicked: {
            stock.add_today()
            stockpic.source =  stock.return_picaddr()
            today.text = stock.return_today()
        }
    }

    Connections {
        target: picup
        onClicked: {
            stock.minus_today()
            stockpic.source =  stock.return_picaddr()
            today.text = stock.return_today()
        }
    }

    Connections{
        target: stock

        onBusysig:{
            busyIndicator.running=indicator
        }
    }



}
