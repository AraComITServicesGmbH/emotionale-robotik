# Emotionale Robotik - Emotionen selbst designt

## Beschreibung

Wir haben einen Server entwickelt, welcher Emotionen anhand von Bildaufnahmen auswertet. Eure Aufgabe ist im Client in der Klasse [OwnRobot](own_robot.py) passende Gesichtsausdrücke mit [Pygame](https://www.pygame.org/docs/) zu generieren. Hierzu könnt ihr euren Code in der Update-Funktion platzieren. Dort stehen euch das aktuelle Foto sowie die dazugehörigen erkannten Emotionen zur Verfügung. Ihr könnte den Parameter update_interval überschreiben, um die Geschwindigkeit der Aktualisierungen zu verändern. Die Auflösung der Raspberry-Pi-Displays beträgt 320 auf 240 Pixel. Viel Spaß beim Coden.

## Einrichtung

Installiert euch [Git](https://git-scm.com/book/de/v2/Erste-Schritte-Git-installieren) und [Python](https://www.computerwoche.de/a/wie-sie-python-richtig-installieren,3548847) auf eurem Rechner. [Klont](https://docs.github.com/de/repositories/creating-and-managing-repositories/cloning-a-repository) euch die Repository auf euer lokales System. Die IP-Adresse des Servers müsst ihr in der [config.py](config.py) hinterlegen. Vor der ersten Nutzung muss die [install.sh](./scripts/install.sh) Datei unter Linux oder die [install.bat](./scripts/install.bat) Datei unter Windows ausgeführt werden. Der Client selbst lässt sich anschließend durch Ausführen der [start_client.sh](./scripts/start_client.sh) Datei unter Linux oder die [start_client.bat](./scripts/start_client.bat) Datei unter Windows im root-Verzeichnis starten.

## Vorhandene Emotionen

- Disgust: Ekel
- Neutral: Neutral
- Sadness: Traurigkeit
- Happiness: Freude
- Fear: Furcht
- Contempt: Verachtung
- Anger: Wut
- Surprise: Überraschung
