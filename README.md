# Deep-Learning-Circle/Square-Recognition-AI (AItRCaS)
AI (programmed with Python, Tensorflow and Keras) to recognize Circles and Squares AItRCaS 

Von Konstantin Reuschenbach 12g und Cyrus Szablowski 12g

![Keras-Logo](images/keras-logo.png)


## Startguide
Zunächst ist Der Code der KI ist in aitdc.py abgespeichert (kann man sich mit Python IDLE anzeigen lassen).
Ausführen lässt sich der Skript in einem virtuellen Enviroment von Anaconda. 
Also muss Anaconda zuerst installiert werden, die Libraries lassen sich über den Anaconda Launcher installieren. Die von uns genutzten Dependencies für keras und tensorflow sind in der req.txt Datei aufgelistet.
Die Bilder mit dem Validationset und Testsset sind in data.zip gespeichert.
Der extrahierte Data Ordner muss mit aitdc.py und den Testbildern in einem Ordner sein. Die Testbilder sind im Ordner Test gespeicher.
Die Prediction läuft über output.py. Das Bild zum Aufbau unseres neuronalen Netzwerks haben wir mit utility.py erstellt, das Bild trägt den Namen model.png. In Model8.zip ist unser trainiertes neuronales Netzwerk gespeichert. 
Alle genannten Dinge in einem Ordner abspeichern, ansonsten verändern der Verzeichnisse.


## Warum wir uns für die Programmierung von einer KI entschieden haben:

KI´s sorgen für Effizienz und Perfektion bei deren Anwendung, bereits im Industriebereich sind deren Einsatzmöglichkeiten sind sehr divers und etabliert. Im generellen spricht man von neuronalen Netzwerken, wobei der Name an ein menschliches Gehirn anspielt. Durch das Lernen bestimmter Dinge können KI´s unsere Kapazitäten übersteigen. Solche neuronale Netzwerke die bestimmte Strukturen wiedererkennen sollen und dann bsp. einen Wert angeben, mit welcher Wahrscheinlichkeit ein Kreis, oder ein Viereck vorhanden ist nennt man functional neural networks.
Bei der Bilderkennung, autonomen Fahren, Ki gesteuertes Computer spielen, personalisierter Werbung, Deepfakes, in der Medizin, in der Erzeugung von Kunstwerken, das Musizieren und vielen weiteren Anwendungsgebieten werden KI´s eingesetzt. Weil KI´s in unserer Gesellschaft eine immer größere Bedeutung finden, haben wir uns entschieden mehr darüber zu lernen. Vorab ist noch zu betonen, dass KI´s nicht auf neuronalen Netzwerken basieren müssen (neuronales Netzwerk ≠ KI). 


### [Arbeitsblog](https://github.com/LordKonstantin007/Arbeitsblog-)

### 1. Programmauswahl
### 2. Programme Installieren
### 3. Erlernen der Grundlagen von KI's
### 4. Aufbau und Code der KI
### 5. Trainieren einer KI
### 6. Datenaufbereitung
### 7. Daten Erratung und Analysing
### 8. Fazit
### 9. Quellen








## 1. Programmauswahl
Zunächst haben wir uns gründlich informiert, welche Programme für das Programmieren von Ki's die einfachste Benutzung haben.

Keras ist eine weitverbreitete Open-Source-Library im Bereich Machine Learning, die sich über den Anaconda launcher mit Leichtigkeit installieren lässt. Die zentralen Merkmale der Bibliothek sind Einfachheit, Erweiterbarkeit und Modularität. Neuronale Netze können mit den bereitgestellten Lösungen erstellt und konfiguriert werden, ohne dass man sich detailliert mit den zugrunde liegenden Backends beschäftigen muss. TensorFlow unterstützt Keras direkt in seiner Kernbibliothek. 

Als Alternative gibt es Python und Scikit-learn. Wir haben uns letzten Endes für Keras und Tensorflow entschieden. Tensorflow ist ein Framework fürs programmieren von KI´s mit Python und C++. Erstellt wurde es von Google und es wird auch von allen Google Diensten verwendet. Fakt ist, dass diese Library unerlässlich für das Programmieren von KI´s ist.

## 2. Programme Instalieren
Das Installieren war leider eine der schwersten Herausforderungen und auch __sehr__ zeitaufwendig. Die verschiedenen Installationswege verwirrten und die Einrichtung von Python fehlte. Man ist immer wieder auf Probleme gestoßen. Über den Python Addon Installer Anaconda lief die Installation mehr oder wenig reibungslos. Das Nutzen der Software innerhalb des Schulunterrichts, war leider anfangs nicht möglich, bzw. das Ausführen des Codes, weil  essentielle Teile der KI sich im Programm nicht aufbauen ließen.
Später haben wir den Fehler behoben. Am Pc zuhause war das coding jedoch immer vollständig möglich. Nebenbei haben wir Befehle für cmd gelernt.
https://www.anaconda.com/distribution/

## 3. Erlernen der Grundlagen von KI's
Die Kenntnis über die Funktionsweise und der Aufbau von Ki´s waren auch von großer Bedeutung, um überhaupt mit dem Programmieren anzufangen. Über Tutorials auf YouTube und Internetadressen konnte viel Wissen mitgenommen werden.
Aufgelistet werden diese in unseren Quellen.

## 4. Aufbau und Code der KI
In unserem Fall programmieren wir ein CNN (Convolutional Neural Network). Diese sind insofern sinvoll, dass sie in der Lage sind Teile/Features auf einem Bild durch bestimmte Filter zu erkennen. Deswegen werden CNNs insbesonders für Bilderkennung verwendet. 

*Ein Convolutional Neural Network (auch „ConvNet“ genannt) ist in der Lage, Input in Form einer Matrix zu verarbeiten. Dies ermöglicht es, als Matrix dargestellte Bilder (Breite x Höhe x Farbkanäle) als Input zu verwenden. Ein normales neuronales Netz z.B. in Form eines Multi-Layer -Perceptrons (MLP) benötigt dagegen einen Vektor als Input, d.h. um ein Bild als Input zu verwenden, müssten die Pixel des Bildes in einer langen Kette hintereinander ausgerollt werden (Flattening). Dadurch sind normale neuronale Netze z.B. nicht in der Lage, Objekte in einem Bild unabhängig von der Position des Objekts im Bild zu erkennen. Das gleiche Objekt an einer anderen Position im Bild hätte einen völlig anderen Input-Vektor.*
Quelle:https://jaai.de/convolutional-neural-networks-cnn-aufbau-funktion-und-anwendungsgebiete-1691/

Ein Input ist beispielsweise ein Bild, Video oder eine Audidatei. Frequenz und Pixel lassen sich durch Encoder in Zahlen darstellen. Mit diesen Zahlen wird im Endeffekt gerechnet. Eine KI kann man sich auch in Form einer komplizierten mathematischen Funktion vorstellen. Man gibt etwas in die Funktion hinein und bekommt etwas heraus. Das Ergebnis wird Im Output der KI angegeben. Zwischen Input und Output verbirgt sich die Struktur des neuronalen Netzwerks. Diese Struktur lässt sich vergleichen mit einem menschlichen Gehirn. Neuronale Netzwerke bestehen aus verschiedenen Schichten (Layern). Diese besitzen eine Tiefe (Depth), deswegen spricht man auch vom Deep learning. Um die Ki´s effektiv trainiern zu können, verwendet man Aktivierungsfunktionen.

Die in unserem Fall wichtigen Layer sind: 
- Convolutional Layer
- Max Pooling Layer
- Flattening 
- Dense Layer (Densely connected Layer)

Dazwischen verbergen sich weitere Funktionen wie
- ReLU (rectified linear unit)
- Sigmoid
- Softmax

### Convolutional Layer
Sie können bestimmte Eigenschaften von Bildern wiedererkennen. Dazu werden die Pixel in Zahlen umgewandelt, danach werden die Zahlen mit einem Filter (bzw. einer Feature Map) skalarmultipliziert. Die Ergebnisse werden zusammen in einer neuen Matrix gespeichert.

![Conv Layer](images/Convlayer.png)


### Max Pooling Layer
Diese Schicht reduziert die Datenmengen auf die Hälfte der vorherigen Größe (bei 2x2 Maxpooling), dabei werden nur das größte Ergebnis aus einem 2x2 Feld übernommen. Grund für die Verwendung ist die relevantesten Signale an die nächsten Schichten weiter zu geben, den Inhalts abstrakter zu machen und die Anzahl der Parameter eines Netzes zu reduzieren.

![Max Pooling Layer](images/MaxpoolSample2.png)

### Dense Layer und Flattening
Beim Flattening Layer (Fully Connected Layer oder Dense Layer) handelt es sich um eine normale neuronale Netzstruktur, bei der alle Neuronen mit allen Inputs und allen Outputs verbunden sind. Um den Matrix-Output der Convolutional- und Pooling-Layer in einen Dense Layer speisen zu können, muss dieser zunächst ausgerollt werden (flattening). Die Output-Signale der Filter-Schichten sind unabhängig von der Position eines Objektes, daher sind zwar keine Positionsmerkmale mehr vorhanden, dafür aber ortsunabhängige Objektinformationen.
Diese Objektinformationen werden in einen oder mehrere Fully Connected Layer eingespeist und mit einem Output-Layer verbunden, welcher zB. genau die Anzahl von Neuronen besitzt, die der Anzahl der verschiedenen zu erkennenden Klassen entspricht.
![Flattening](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/73_blog_image_2.png)

### ReLU (rectified linear unit)
Diese Aktivierungsfunktion ist wichtig für den Nomalisierungs Prozess. Aktivierungsfunktionen können bestimmte Neuronen mit denen sie weiterverknüpft sind aktivieren (1) und deaktivieren (0). Hierbei werden negative Werte normalisiert, bzw. wird das Signal des Outputs so verändert, sodass eventuell das folgende Neuron deaktiviert wird. Zahlen größer als 0 entsprechen ihrem Eingabewert.
f(x) = max(0,x)

![ReLU-Function](images/ReLU-Function.png)

### Sigmoid function
Bei der Sigmoidfunktion wird der Output eines Neurons so verändert, dass dieser einen Wert zwischen (1) und (0) besitzen. Je näher der Wert an der (1) grenzt, desto eher wird das nächste Neuron aktiviert und entgegengesetzt in der Nähe von (0) deaktiviert.       

![Sigmoid-function-2 svg](https://user-images.githubusercontent.com/54355257/68784163-7214ed80-063c-11ea-9223-1ac9861a4f11.png)

### Softmax function
Die Softmaxfunktion benutzt man wenn man eine Klasssifikation durchführen, wobei mehr als zwei Klassen vorhanden sind. Zum Beispiel hat man vier Klassen (a,b,c,d). Jede Klasse ist für ein bestimmtes Ergebniss vorhanden. Die Wahrscheinlichkeit die für eine Klasse steht muss (1=100%=Aktiviert) anzeigen und alle anderen 3 Klassen (0=0%=Deaktiviert). Unteranderem unterlaufen der KI Fehler, plötzlich weisen mehrere Klassen den Wert (1) auf. Durch die Softmax funktion wird bestimmt, dass nur ein Neuron aktiviert werden soll um die überflüssigen zu deaktivieren. Damit wird gegeben, das das Ergebniss nur zur einer Klasse definiert wird. Somit können KI's bspw. in Ziffern von 0-9 (MNIST Dataset: num_classes = 10) unterscheiden.

![Softmax-Function](images/Softmax-Function.png)
![Softmax-Function2](https://qph.fs.quoracdn.net/main-qimg-fda2f008df90ed5d7b6aff89b881e1ac.webp)

### MNIST Dataset
Das MNIST Dataset ist eine Datenbank, in welcher sehr viele Bilder von handgeschriebenen Ziffern gespeichert sind.
Diese besitzen die Größe von 28x28 Pixeln. Vorm Beginn des Cooden von KI´s ist es so gesehen eine Pflicht sich das MNIST Dataset anzuschauen bzw. eine KI damit zu trainieren. Dadurch lernt man auch praktisch sehr viel.

![MNIST](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)
https://keras.io/datasets/

### Importieren der Libraries bzw. der Werkzeuge die wir für das Bauen der KI brauchen. 

Als erstes brauchen wir den ImageDataGenerator, dieser erstellt mehrere Daten/Bilder aus einem Bild.
Wir brauchen außerdem ein Sequentialmodel und kein functional API Model. 
Außerdem brauchen wir Aktivierungsfunktionen damit das neuronale Netzwerk Neuronen aktivieren und deaktivieren kann. 
Dropout ist wichtig damit zufällige Neuronen deaktiviert werden. Das ist wichtig, damit das Neuronale Netzwerk nicht overfitted/überangepasst ist.
Flatten ist notwendig um unsere 2D Daten in 1D Arrays zu konvertieren, denn nur mit Ihnen kann die KI rechnen. 
Dense wird verwendet um ein Hidden Layer an unser Output Layer anzuhängen.
Außerdem verwenden wir natürlich Keras und Numpy. Numpy verwenden wir um unsere Arrays zu manipulieren, oder damit wir einfach unsere Ergebnisse in Arrays anzeigen können.
Zu guter letzt brauchen wir keras.preprocessing import image, damit unsere Bilder importieren und vorverarbeiten können.

```
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K 
import numpy as np
from keras.preprocessing import image
``` 


### Image Preprocessing:

Als erstes müssen die Bilder unseres Datasets gelesen werden können. Dazu geben wir die Größe (width & height) des Bildes und die Directories für das Trainings- und Validationset als Variable an.
Außerdem definieren wir die Größe des Datensatzes, also wieviele Bilder wir für das Trainieren bereitstellen. Normalerweise gibt man 
KI´s viel mehr Bilder als es bei uns der Fall mit 1000 sind. Doch je größer so ein Datensatz ist, desto länger dauert auch das Training, dafür ist die KI im Erkennen noch präziser.
Ein kompletter Trainingsdurchlauf aller Input-Daten wird dabei jeweils als Epoche bezeichnet.
Je öfter man eine KI mit dem selben Datensatz trainiert, also je größer die Epochenanzahl, desto besser passt sich die KI der Bilder an. Dabei steigt die Genauigkeit und es sinkt die Lossrate. Epochen lassen sich zusätzlich in Batches einteilen. 
Wenn alle Batches das neuronale Netz ein Mal durchlaufen haben, ist eine Epoche vollendet. Unsere batch_size ist ein Hyperparameter der beim Trainieren die Anzahl von Samples bestimmt die durch die KI laufen, bevor ihre Parameter (Biases, Weights) geupdated werden.

``` 
img_width, img_height = 200,200

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 15
batch_size = 20
``` 

Als nächstes müssen wir klarstellen, dass unsere Bilder im Input die richtige Form haben (channels, height, width/3x200x200) oder ( height, width,channels/200x200x3). 
Mit Channels sind bei uns die RGB Farben gemeint. Da alle Farben von Pixeln durch drei RGB-Werte definiert sind, sprechen wir von Bildern mit 3 Farbkanälen/Channels. 
Mit train_datagen = ImageDataGenerator erstellen wir ein noch größeres Dataset fürs Trainieren,dabei entstehen weit mehr als unseren ursprünglichen 1000 Bildern, diese sind jedoch nicht aufrufbar, es gibt aber Tools, mitwelchen man sich die von KI´s verarbeiteten Bilder anschauen kann.
Mit rescale Skalieren/Multiplizieren wir die Daten um den Faktor 1/255, bevor wie sie weiter verarbeiten. Die shear_range gibt die Scherintensität an also der Scherwinkel gegen den Urzeigersinn in Grad, mit welchem das Bild verzogen wird (siehe Scherung bei der Geometrie). Die zoom_range steht für das zufällig auftretende Reinzoomen von Bildern.
horizontal_flip dient zum zufälligen Spiegeln der Hälfte der Bilder in horizontaler Richtung. Beim Testen wird auch ein weiterer Datensatz erstellt der die Bilder nur Neuskaliert, bzw. mit dem Faktor 1/255 multipliziert.

``` 
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

``` 
Hier erstellen wir ein Training und ein Validation Data Generator. Diese Beiden Generator führen unser Anweisungen für das Image Processing in train_datagen und test_datagen aus. Dafür muss nochmals das Verzeichnis angegeben werden, sowie die Größe des Bildes, die batch size an.
Unsere Daten lassen sich in einem 1D numpy array umschreiben, deshalb verwenden wir class_mode='binary'.

``` 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height), 
    batch_size=batch_size,
    class_mode='binary')
    
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
``` 



	



### Aufbau des ConvNets
Der Hauptkörper unserer Struktur lässt sich in zwei Teile unterteilen. Am Anfang steht das Convolutioning im Vordergrund bzw. das Feature Extracting und am Ende das Klassifizieren. Zunächst verwnden wir ein Conv Layer, die ReLU Aktivierungsfuktion und ein Maxpooling-Layer. Diese Aufeinanderreihung der Schichten/Operationen machen im wesentlichen die Struktur des CNNs aus.

#### Feature Extracting
``` 
model = Sequential()
``` 
Als erstes erstellen wir ein Objekt mit dem namen model.
Durch model.add fügen wir dem Objekt, unserem neuronalen Netzwerk eine Struktur zu. Als erstes nutzen wir ein Conv Layer, um bestimmte Features aus den Bildern extrahieren. Für jedes Bild werden 32 Features extrahiert. Die Größe der Suchenden Filter Matrizen haben dabei die Größe von 3x3, zusätzlich wird nochmals die Größe des Inputs angegeben.
Mit der Aktivierungsfunktion ReLU, normalisieren wir alle negativen Werte, sodass das folgende Neuron deaktiviert wird. 
Darauf wird unser Bild auf die wichtigsten Informationen reduziert, indem das Bild mit MaxPooling verkleinert wird. Für alle 2x2 Felder im Bild werden die größten Werte in einem neuen kleineren Bild übertragen. Diese Schritte wiederholt die KI drei mal, wobei beim dritten Mal nach 64 Features auf dem Bild gesucht wird.

``` 
model.add(Conv2D(32, (3, 3), input_shape=input_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
``` 

#### Classification
Aus unseren vielen kleinen 2 dimensionalen Feature Maps, erstellen wir durch Flatten 1 dimensionale Bilder. Danach verwenden wir Dense mit den Wert mit der Anzahl von vorhandenen Featuremaps pro Bild (64).
model.add(Dense(1)) wird zu unserem letzten Output, dieser gibt einen Wert an eine Sigmoid Funktion, diese gibt einen Output von 0 oder 1. Also ob es sich um ein Kreis oder Viereck handelt.
 
``` 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
``` 
![CNN](https://jaai.de/wp-content/uploads/2018/02/Typical_cnn.png)

## 5. Trainieren einer KI 
Für das Trainieren (bei unserem Fall dem supervised learning) einer KI nutzt man Backpropagation. Aber warum überhaupt das Training. Am Anfang ist eine neuronales Netzwerk auf nichts spezialisiert, das Bedeutet das die KI nicht einer Funktion nachgehen kann, weil sie Dinge die sie erkennen soll nicht erkennt.
Deswegen ist das Trainieren von KI´s wichtig. Jedoch muss aufgepasst werden, dass die KI nicht overfitted oder underfitted ist. Das bedeutet, dass die KI nicht immer das selbe Bild sieht und eine richtige Antwort gibt, sondern das verschiedene wesentliche Strukturen von Bildern erkannt werden. Somit ist die KI auf eine bestimmte Erkennung spezialisiert und nicht auf ein bestimmtes Bild.
Die zu veränderende Werte in der KI sind Biases und Weights.
Das neuronale Netzwerk hat eine hohe LOSSRATE (Fehlerquote), doch dieses kann man durch das Training möglichst erniedrigen, sodass die Genauigkeit (accuracy) steigt. Gleichzeitig versucht man den Losswert möglichst gering zuhalten, dafür gibt es verschiedene Optimierungmethoden. (Optimizer)

![Loss-and-Accuraccy](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/11/Line-Plots-of-Cross-Entropy-Loss-and-Classification-Accuracy-over-Training-Epochs-on-the-Two-Circles-Binary-Classification-Problem.png)

### Biases und Weights
Ein Gewicht repräsentiert die Stärke der Verbindung zwischen Neuronen. Wenn das Gewicht von Neuron 1 zu Neuron 2 größer ist, bedeutet dies, dass Neuron 1 einen größeren Einfluss auf Neuron 2 hat. Ein Gewicht verringert die Wichtigkeit des Eingabewerts. Gewichte nahe Null bedeuten, dass durch Ändern dieses Eingangs der Ausgang nicht geändert wird. Negative Gewichte bedeuten, dass durch Erhöhen dieser Eingabe die Ausgabe verringert wird. Ein Gewicht bestimmt deshalb, wie stark die Eingabe die Ausgabe beeinflusst. Diese Parameter bestimmen unteranderem, ob ein Folge Neuron aktiviert wird.

Jedes Neuron besitzt ein Bias, somit gibt es eine große Anzahl von beeinflussenden Zahlen im neuronalen Netzwerk. Die Werte von den Biases werden genau wie die Weights durch die Optimizer, während des Backpropagation Prozesses geupdatet. Sie können zusätzlich bestimmen, ob ein Folgeneuron aktiviert/deaktiviert wird. Generell sind Biases insofern nützlich, dass sie die Flexibilität der KI beim Erkennen von Objekten erhöhen. Durch Normalisation können zum Beispiel ein Neuron deaktiviert werden, doch dadurch das der Output dieses Neurons ein Bias besitzt könnte er ein Folge Neuron aktivieren. Biases und Weights lassen sich jedoch in unserem Fall nicht Manuell für ein Neuron verändern, dies geschieht automatisch im Lern Prozess.

![Biases+Weights](images/Bias+Weights.png)

Beim Trainieren öffnet sich ein Ladebalken für jede Epoche, wobei zusätzlich die Genauigkeit und der Loss angegeben wird.
Nachdem das Trainiern fertig war, ließ sich die Funktion der KI ausprobieren.
![Training](images/Trained.PNG)
__Wichtig!__ Erst nach dem Trainieren den Test Ordner in den Validationset Ordner packen!

### Optimizer
Während des Trainings optimieren und ändern wir die Parameter (Gewichte) unseres Modells, um unseren Loss zu minimieren und unsere Vorhersagen so korrekt wie möglich zu machen. Aber wie genau macht man das? Wie und wann ändernt man die Parameter eines Modells?

Hier kommen Optimierer ins Spiel. Sie verknüpfen die Loss funktion und die Modellparameter, indem sie das Modell als Reaktion auf die Ausgabe der Verlustfunktion aktualisieren. Einfacher ausgedrückt: Optimierer formen und formen das Modell in die genaueste Form, indem sie den Wert der Gewichten und Biases verändern. Der Optimizer regelt den Trainingsprozess, damit die Gewichte so verändert werden, dass der Loss-Wert sinkt und die Genauigkeit steigt.

Die Hyperparameter der KI kann man sich auch in einem Hyperdimensionalen Raum vorstellen (feature Space). Die Optimizer versuchen zu Beginn am Training von einem Maxium auf ein Minimum zu treffen. Den schnellsten/steilsten Weg bestimmen hier die verschiedenen Optimizer, sodass die KI schnellst möglich optimiert wird.


![Feature Space](https://blog.paperspace.com/content/images/2018/05/convex_cost_function.jpg)
Wir haben den Optimizer ADADELTA verwendet.
Das Verfahren von Adadelta passt sich dynamisch mit der Zeit an, wobei nur Informationen erster Ordnung verwendet werden, und weist einen minimalen Rechenaufwand auf, der über den stochastischen Gradientenabfall hinausgeht. Diese Optimizer sind ebenfalls durch komplizierte mathematischen Formeln definiert.

``` 
model.compile(Loss='binary_crossentropy',
      optimizer='adadelta',
      metrics=['accuracy'])
``` 
## 6. Datenaufbereitung
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
![Dataset](images/Dataset.PNG) 
           



## 7. Daten Erratung und Analysing

Wir haben die Daten Erratung in einer seperat abgespeicherten Python-Datei gespeichert (output.py).
Somit müssen wir das Model erneut laden, sowie die notwendigen Libraries.
Für das Erraten eines Bilds können wir eines vom Validationset laden, dazu wird die Größe des Bildes nochmals angegeben
Zusätzlich muss das Bild in ein Array konvertiert werden, hierzu verwenden wir Numpy.
```
import numpy as np

from keras.preprocessing import image
from keras.models import load_model

model = load_model('model.h5')

img_pred = image.load_img('data/validation/test/x.png', target_size = (200, 200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
```
Das oben zu erattende Bild wird als Ergebnis definiert, dieses Ergebnis wird in der Konsole angezeigt.
Es sollte zwischen 0 und 1 liegen. Wenn das Ergebnis gleich 1 ist, soll die Konsole sagen, dass es es sich um ein Viereck handelt.
Ist das Ergebnis nicht 1, so handelt es sich um ein Kreis.

```
rslt = model.predict(img_pred)
print (rslt)
if rslt[0][0] == 1:
     prediction = "Viereck"
else:
     prediction = "Kreis"    
print (prediction) 
```

#### ![Test1](images/test1.png) Testbild1 erkannt als Kreis
#### ![Test2](images/test2.png) Testbild2 erkannt als Kreis
#### ![Test3](images/test3.png) Testbild3 erkannt als Viereck
#### ![Test4](images/test4.png) Testbild4 erkannt als Viereck
#### ![Test5](images/test5.png) Testbild5 erkannt als Viereck 
#### ![Test6](images/test6.png) Testbild6 erkannt als Viereck
#### ![Test7](images/test7.png) Testbild7 erkannt als Kreis
#### ![Test8](images/test8.png) Testbild8 erkannt als Viereck
#### ![Test9](images/test9.png) Testbild9 erkannt als Kreis
#### ![Test10](images/test10.png) Testbild10 erkannt als Kreis
#### ![Test11](images/test11.png) Testbild11 erkannt als Kreis

Weil die Trainierte KI alle Bilder im Validationset 100% richtig bestimmt, haben wir uns überlegt ein paar eigene Testbilder zu machen und diese durch die KI zu erraten. Besonders interessant ist das Ergebnis für Testbild 5,8,9,10,11.
Bild 9 ähnelt einem Kreis eher als Bild 5, weil es zwei senkrechte Ecken besitzt. Dennoch wird 5 als Viereck und 9 als Kreis erkannt.
Bild 10 lässt sich eigenlich auch einfach erklären. Dadurch, dass alle unsere Bilder Schwarze Objekte auf weißem Hintegrund waren, ist unsere KI auf die äußeren Katen des Kreisen spezialisiert, somit spielt es keine Rolle, dass ein weißes Viereck im Bild vorhanden ist.
Mit Testbild 11 lässt sich die KI einfach perfekt zusammenfassen:
Die Dinge mit der wir sie trainiert haben kann sie gut erkennen, aber Kreise und Vierecke im allgemeinen nicht! Grund dafür ist, dass unsere Bilderdatenbank zu klein ist. Gäbe man der KI 10 000 oder 100.000 Bilder ist die Wahrscheinlichkeit, dass eine KI etwas richtig erkennt sehr hoch. Dabei müssen aber Werte wie epochs und batches richtig verändert werden, sodass die KI eine perfekte Angepasstheit bekommt. Dadurch würde aber auch der Trainingsprozess viel länger dauern, deshalb haben wir uns auf 1000 Bilder beschränkt.

## 8. Fazit
Das Projekt hat viel Spaß gemacht und Cyrus überlegt weitere KI´s eventuell, sofern die Zeit es erlaubt zu programmieren. Nicht nur Spaß, sondern auch viel Erfahrung hat dieses Projekt mit sich getragen. Auch das Programmieren einer KI mit einer wirklich eigenen Herausforderung wäre ein Cooles Ziel für ein anders Projekt. Dieses Projekt diente aber mehr dazu Grundverstänis zu neuronalen Netzwerken zu gewinnen.

## 9. Quellen

### Download Anaconda
#### https://www.anaconda.com/distribution/

### Keras 
#### https://keras.io/
#### https://keras.io/datasets/
#### https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#### https://keras.io/models/sequential/
#### https://keras.io/layers/convolutional/
#### https://keras.io/layers/core/
#### https://keras.io/layers/pooling/
#### https://keras.io/preprocessing/image/
#### https://keras.io/visualization/
#### https://keras.io/optimizers/
#### https://keras.io/applications/


### Andere Links
#### https://stackoverflow.com/questions/8155060/moving-specific-files-in-subdirectories-into-a-directory-python
#### https://www.kaggle.com/smeschke/four-shapes
#### https://github.com/hatemZamzam/Cats-vs-Dogs-Classification-CNN-Keras-/blob/master/cnn.py
#### https://www.tensorflow.org/
#### https://jaai.de/convolutional-neural-networks-cnn-aufbau-funktion-und-anwendungsgebiete-1691/
#### https://www.quora.com/How-does-softmax-function-work-in-AI-field
#### https://hackernoon.com/everything-you-need-to-know-about-neural-networks-8988c3ee4491
#### https://algorithmia.com/blog/introduction-to-optimizers



### You Tube Playlist
#### https://www.youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
#### https://www.youtube.com/watch?v=RznKVRTFkBY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL
#### https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
#### https://www.youtube.com/watch?v=-7scQpJT7uo
#### https://www.youtube.com/watch?v=qx8l-LmdgEk
#### https://www.youtube.com/watch?v=ILsA4nyG7I0
#### https://www.youtube.com/watch?v=HMcx-zY8JSg
#### https://www.youtube.com/watch?v=oOSXQP7C7ck
#### https://www.youtube.com/watch?v=FmpDIaiMIeA
#### https://www.youtube.com/watch?v=XNKeayZW4dY&t=894s


### Coole Videos zu KI´s
#### https://www.youtube.com/watch?v=UWxfnNXlVy8&t=666s
#### https://www.youtube.com/watch?v=qv6UVOQ0F44
#### https://www.youtube.com/watch?v=zIkBYwdkuTk
#### https://www.youtube.com/watch?v=UWxfnNXlVy8






