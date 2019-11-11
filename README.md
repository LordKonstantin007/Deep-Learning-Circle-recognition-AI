# Deep-Learning-Circle/Square-Recognition-AI (AItRCaS)
AI (programmed with Python, Tensorflow and Keras) to recognize Circles and Squares AItRCaS 

![Keras-Logo](images/keras-logo.png)

## Warum wir uns für die Programmierung von einer KI entschieden haben:

KIs übernehmen Momentan eine Menge Arbeit im Industriebereich und deren Einsatzmöglichkeiten sind sehr divers.
Im generellen spricht man von Neuronalen Netzwerken, die wie ein Menschliches Gehirn durch das Lernen bestimmte Aufgaben übernehmen können. Solche Neuronale Netzwerke die Bestimmte Strukturen wiedererkennen sollen und dann bsp. einen Wert angeben, mit welcher Wahrscheinlichkeit, wie viele Kreise zu erkennen sind oder ob überhaupt welche vorhanden sind nennt man functional neural networks.
Bilderkennung, Deepfakes und viele Aufgaben die KIs künftig übernehmen könnten. Weil KIs in unserer Gesellschaft immer größere Bedeutung finden, haben wir uns entschieden mehr darüber zu lernen.

### 1. Programmauswahl
### 2. Programme Instalieren
### 3. Erlernen der Grundlagen von KI's
### 4. Aufbau der KI
### 5. Datenaufbereitung
### 6. Training
### 7. Training Process and Analysing

## 1. Programmauswahl
Zunächst haben wir uns gründlich informiert, welche Programme für das Programmieren von Ki's die einfachste Benutzung haben.

Keras ist eine Open-Source-Library zum einfachen Programmieren von neuronalen Netzwerken, die sich über den Anaconda launcher mit leichtigkeit installieren lässt. Die zentralen Merkmale der Bibliothek sind Einfachheit, Erweiterbarkeit und Modularität. Neuronale Netze können mit den bereitgestellten Lösungen erstellt und konfiguriert werden, ohne dass man sich detailliert mit den zugrunde liegenden Backends beschäftigen muss. TensorFlow unterstützt Keras direkt in seiner Kernbibliothek. 

Als Alternative gibt es Python, Scikit-learn und Docker. Wir haben bspw. Docker ausprobiert, doch Keras erwies sich aus meheren Gründen als einfacher. Auch das erstellen von virtuellen Envirements ist in Anaconda viel leichter als das Nutzen von Daemons bei Docker, zusätzlich ist der Code in Keras viel ordentlicher. Tensorflow ist ein Framework fürs programmieren von KIs mit Python und C++. Erstellt wurde es von Google und es wird auch von allen Google Diensten verwendet. Letztenendes haben wir Tensorflow nicht in unserem Script gebraucht, dennoch wird es auch bei Scikit-learn und Docker verwendet und spielt für das Programmieren von KIs eine zentrale Rolle. 

## 2. Programme Instalieren
Das Installieren war leider einer der Schwersten Herausforderungen und auch __sehr__ Zeitaufwendig. Die verschiedenen Installationswege verwirrten und die Einrichtung von Python fehlte. Man ist immer wieder auf Probleme gestoßen. Über den Python Addon Installer Anaconda lief die Installation mehr oder wenig reibungslos. Das Nutzen der Software innerhalb des Schulunterrichts, war leider nicht möglich, bzw. das Ausführen des Codes, weil  essentielle Teile der KI sich im Programm nicht aufbauen ließen. Am Pc zuhause war das coding jedoch vollständig möglich. Nebenbei haben wir Befehle für cmd gelernt.
https://www.anaconda.com/distribution/

## 3. Erlernen der Grundlagen von KI's
Die Kenntnis über die Funktionsweise und der Aufbau von Kis waren auch von großer Bedeutung, um überhaupt mit dem Programmieren anzufangen. Über Tutorials auf YouTube, Wissenschaftlichen Papers und Internetadressen konnte viel Wissen mitgenommen werden.
Aufgelistet werden diese in unseren Quellen.

## 4. Aufbau der KI
In unserem Fall programmieren wir ein CNN (Convolutional Neural Network). Diese sind insofern sinvoll, dass sie in der Lage sind Teile auf einem Bild durch bestimmte Filter zu erkennen. Deswegen werden CNNs insbesonders in der Bilderkennung verwendet. 

*Ein Convolutional Neural Network (auch „ConvNet“ genannt) ist in der Lage, Input in Form einer Matrix zu verarbeiten. Dies ermöglicht es, als Matrix dargestellte Bilder (Breite x Höhe x Farbkanäle) als Input zu verwenden. Ein normales neuronales Netz z.B. in Form eines Multi-Layer -Perceptrons (MLP) benötigt dagegen einen Vektor als Input, d.h. um ein Bild als Input zu verwenden, müssten die Pixel des Bildes in einer langen Kette hintereinander ausgerollt werden (Flattening). Dadurch sind normale neuronale Netze z.B. nicht in der Lage, Objekte in einem Bild unabhängig von der Position des Objekts im Bild zu erkennen. Das gleiche Objekt an einer anderen Position im Bild hätte einen völlig anderen Input-Vektor.*
Quelle:https://jaai.de/convolutional-neural-networks-cnn-aufbau-funktion-und-anwendungsgebiete-1691/

Ein Input ist Beispielsweise ein Bild, Video oder eine Audidatei. Frequenz und Pixel lassen sich durch Encoder in Zahlen darstellen. Mit diesen Zahlen wird im Endeffekt gerechnet. Eine KI kann man sich auch in Form einer komplizierten mathematischen Funktion vorstellen. Man gibt etwas in die Funktion hinein und bekommt etwas heraus. Das Ergebnis wird Im Output der KI angegeben. Zwischen Input und Output verbirgt sich die Struktur des Neuronalen Netzwerks. Diese Struktur lässt sich vergleichen mit einem menschlichen Gehirn. Neuronale Netzwerke bestehen aus verschiedenen Schichten (Layern). Diese besitzen eine Tiefe (Depth), deswegen spricht man auch vom Deep learning. Um die Kis effektiv trainiern zu können, verwendet man Aktivierungsfunktionen.

Die in unserem Fall wichtigen Layer sind: 
- Convolutional Layer
- Max Pooling Layer
- Flattening Layer
- Dense Layer (Densely connected Layer)

Dazwischen verbergen sich weitere Funktionen wie
- ReLU (rectified linear unit)
- Sigmoid
- Softmax







__ReLU (rectified linear unit)__
Diese Aktivierungsfunktion ist wichtig für den Nomalization Process. Aktivierungsfunktionen können bestimmte Neuronen mit denen sie weiterverknüpft sind aktivieren (1) und deaktivieren (0). Hierbei werden negative Werte normalisiert, bzw. wird das Signal des Outputs so verändert, sodass das folgende Neuron deaktiviert wird. Zahlen größer als 0 bleiben gleich.
f(x) = max(0,x)

__Sigmoid function__
Diese Aktivierungsfunktion










## 5. Datenaufbereitung
Die Datenaufbereitung ist wenn man Software und Verständnis über das Programmieren hat, auch eine große Herausforderung.
Man brauch nähmlich ein Dataset. Dieses muss eingeteilt werden. In ein Trainingset und in ein Validationset. Dabei sollte das Validationset ungefähr 20-10% der Größe des Trainingsets entsprechen. Um zufällig diese Anzahl zu transferieren haben wir schnell ein Python Script dafür geschrieben. Somit kann das Trainieren Beginnen.
```
import os
import shutil
import fnmatch

def gen_find(filepat,top):
    print("gen_finding")
    i = 0
    for path, dirlist, filelist in os.walk(top):
        i+=1
        print("outer")
        print(i)
        j = 0
        for name in fnmatch.filter(filelist,filepat):
            j+=1
            print("inner")
            print(j)
            yield os.path.join(path,name)

# Example use
def do():
    print("doing")
    src = './data/train/Vierecke' # input
    dst = './data/validation/Vierecke' # desired location

    filesToMove = gen_find("*.png",src)
    for name in filesToMove:
        splitName = name.split(".png")[0].split('\\')[-1]
        print(splitName)
        numberAsString = splitName
        print(numberAsString)
        number = int(numberAsString)
        
        if number % 10 == 0:
            shutil.move(name, dst)
```
           
https://stackoverflow.com/questions/8155060/moving-specific-files-in-subdirectories-into-a-directory-python

## 6. Trainieren einer KI 
Für das Trainieren einer KI nutzt man Backpropagation. Aber warum überhaupt das Training. Am Anfang ist eine Neuronales Netzwerk auf nichts spezialisiert, das Bedeutet das die KI nicht einer Funktion nachgehen kann, weil sie Dinge die sie erkennen soll nicht erkennt.
Deswegen ist das Trainieren von KIs wichtig. Jedoch muss aufgepasst werden, dass die KI nicht overfitted oder underfitted ist. Das Bedeutet, dass die KI nicht immer das selbe Bild sieht und eine richtige Antwort gibt, sondern das verschiedene wesentliche Strukturen von Bildern erkannt werden. Somit ist die KI auf eine Bestimmte Erkennung spezialisiert und nicht auf ein Bestimmtes Bild.
Die zu Veränderende Werte in der KI sind Biases und Weights.
Das Neuronale Netzwerk hat eine hohe LOSSRATE (Fehlerquote), doch dieses kann man durch das Training möglichst erniedrigen, sodass die Genauigkeit (accuracy) steigt. Gleichzeitig versucht man den Losswert möglichst gering zuhalten, dafür gibt es verschiedene Optimierungmethoden. (Optimizer)














## Daten Erratung
Als erstes Speichern wir die Größe unserer Gewichte in einer .h5 Datei.
```model.save_weights('first_try.h5')```
Für das Erraten eines Bilds können wir eines vom Validationset laden, dazu wird die Größe des Bildes nochmals angegeben
Zusätzlich muss das Bild in ein Array konvertiert werden, hierzu verwenden wir Numpy.
```
img_pred = image.load_img('data/validation/?', target_size = (200, 200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
```
Das oben geladene Bild wird als Ergebnis definiert, dieses Ergebnis wird in der Konsole angezeigt.
Es sollte zwischen 0 und 1 liegen. Wenn das Ergebnis gleich 1 ist, soll die Konsole sagen, dass es es sich um ein Kreis handelt.
Ist das Ergebnis nicht 1, so handelt es sich um ein Viereck.
```
rslt = model.predict(img_pred)
print (rslt)
if rslt[0][0] == 1:
     prediction = "Kreis"
else:
     prediction = "Viereck"    
print (prediction) 
```
Dieses ErratungsPrinzip haben wir uns hiervon abgeschaut. https://github.com/hatemZamzam/Cats-vs-Dogs-Classification-CNN-Keras-/blob/master/cnn.py



