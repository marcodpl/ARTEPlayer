from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
import PyQt5.QtWebEngineWidgets
import os


class Ui_MapWindow(object):
    def setupUi(self, MapWindow):
        MapWindow.setObjectName("MapWindow")
        MapWindow.resize(800, 524)
        self.mainLayout = QtWidgets.QVBoxLayout(MapWindow)
        self.mainLayout.setObjectName("mainLayout")
        self.topNavLayout = QtWidgets.QHBoxLayout(MapWindow)
        self.topNavLayout.setObjectName("topNavLayout")
        self.mediaTab = QtWidgets.QPushButton(MapWindow)
        self.mediaTab.setObjectName("mediaTab")
        self.topNavLayout.addWidget(self.mediaTab)
        self.mapTab = QtWidgets.QPushButton(MapWindow)
        self.mapTab.setMinimumSize(QtCore.QSize(0, 50))
        self.mapTab.setObjectName("mapTab")
        self.topNavLayout.addWidget(self.mapTab)
        self.phoneTab = QtWidgets.QPushButton(MapWindow)
        self.phoneTab.setObjectName("phoneTab")
        self.topNavLayout.addWidget(self.phoneTab)
        self.settingsTab = QtWidgets.QPushButton(MapWindow)
        self.settingsTab.setObjectName("settingsTab")
        self.topNavLayout.addWidget(self.settingsTab)
        self.mainLayout.addLayout(self.topNavLayout)
        self.mapView = QtWebEngineWidgets.QWebEngineView(MapWindow)
        self.mapView.load(QtCore.QUrl.fromLocalFile(
            os.path.join(
                f"{os.path.dirname(__file__)}\\..\\..\\maps_engine",
                "maps_html.html"
            )
        ))
        self.mainLayout.addWidget(self.mapView)

        self.retranslateUi(MapWindow)
        QtCore.QMetaObject.connectSlotsByName(MapWindow)

    def retranslateUi(self, MapWindow):
        _translate = QtCore.QCoreApplication.translate
        MapWindow.setWindowTitle(_translate("MapWindow", "Maps"))
        self.mediaTab.setStyleSheet(_translate("MapWindow", "font-size: 16px;"))
        self.mediaTab.setText(_translate("MapWindow", "🏠 Home"))
        self.mapTab.setStyleSheet(_translate("MapWindow", "font-size: 16px;"))
        self.mapTab.setText(_translate("MapWindow", "🗺️ Maps"))
        self.phoneTab.setStyleSheet(_translate("MapWindow", "font-size: 16px;"))
        self.phoneTab.setText(_translate("MapWindow", "🎵 Media"))
        self.settingsTab.setStyleSheet(_translate("MapWindow", "font-size: 16px;"))
        self.settingsTab.setText(_translate("MapWindow", "⚙️ Settings"))
