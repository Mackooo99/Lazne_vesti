## 📂 Dataset

Zbog ograničenja veličine fajlova na GitHub-u, skupovi podataka se nalaze na Google Drive-u.
Fake.csv i True.csv služe za trening i validaciju modela dok News_2025.csv služi ya testiranje.

- [Fake.csv](https://drive.google.com/file/d/1xuje2pCRVfxIT90d_oRXnN5M7DmcxFp1/view?usp=drive_link)
- [True.csv](https://drive.google.com/file/d/1KrlxW26UvSSiyxDxLuFegoDOVlmzCPHJ/view?usp=drive_link)
- [News_2025.csv](https://drive.google.com/file/d/1lOlMO7vPOvXnzmPmi5l0rRnTxRTWc2jj/view?usp=drive_link)

Preuzmite ih i postavite u odgovarajuće direktorijume pre pokretanja skripti.

Prvo se pokrece kod analyze_dataset.py gde se radi analiza True.csv i Fake.csv i vizuelni prikaz rezultata.

Zatim se pokreće data_preprocessing.py koji sređuje podatke i spaja ih jednu veliku bazu dodavajući jos neke numeričke osobine kasnije korišćenje u treningu.

Na kraju se pokrece logic_reg.py gde se koristi model Logičke regresije.
