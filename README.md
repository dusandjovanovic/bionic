# bionic
Artifficial neural networks implementation. Weather prediction.

## Opis problema

Primena neuronske mreže za **predviđanje vremenske prognoze**. Korisnost neuronskih mreža u predviđanju je i te kako velika, obzirom da mogu da uče na osnovu već prisutnih podataka. Način učenja ogleda se u prilagođavanju veza neuronske mreže tako da se dobije isti rezultat na osnovu drugih podataka. Na primer, za određenu **situaciju** (uzrok) postoji konkretna **posledica** (rezultat) i upravo ovaj odnos se tumači i kodira u vidu podataka. Neuronska mreža pritom uči nelinearnu funkciju koja preslikava uzrok u rezultat.
 
Primer tabele sa podacima koji oslikavaju vremenske uslove. Tabela sadrži pet promenljivih koje opisuju hipotetičke vremenske uslove. Uz pretpostavku da se za svaku promenljivu takođe vezuje lista podataka koja pokriva određeni vremenski period (vremenske serije). Upravo odnos između takvih vremenskih serija predstavlja dinamičku reprezentaciju vremenskih prilika u nekom mestu. Od neuronske mreže se očekuje da **nauči ovu dinamiku**.

![alt text][window]

[window]: images/window.png


## Način rešavanja

### Izbor podataka
Većina API-a za pribavljanje podataka o vremenskim prilikama imaju za parametre:
* Temperaturu (°C)
* Vlažnost vazduha (%)
* Pritisak (mbar)
* Padavine i slično..

### Izbor ulaznih i izlaznih promenljivih
Prvi korak u struktuiranju neuronske mreže svodi se na izbor podataka koji će se dostavljati, pritom uzevši u obzir model predstavljanja podataka. Neuronske mreže rade poput nelinearnih blokova sa predefinisanim ulazima i izlazima, prema tome neophodno je izabrati koju će ulogu imati svaka vremenska promenljiva u jednoj ovakvoj mreži. Drugim rečima, treba odrediti koje će promenljive neuronska mreža da predviđa i korišćenjem kojih ulaznih promenljivih. Za demonstraciju mogućnosti neuronskih mreža u predviđanju vremenskih prilika, za predviđanje (izlaz) je izabrana **prosečna temperatura** u jednom danu, predviđena na osnovu ostalih vremenskih promenljivih (ulazi). *Dakle, ulazne promenljive će biti: padavine, izlaganje suncu, vlažnost vazduha i brzina vetra. Dok je izlazna vrednost koja se predviđa srednja dnevna temperatura, na osnovu ulaznih promenljivih.*

### Filtriranje podataka
Veoma često se dešava da podaci predati neuronskoj mreži imaju manjkavosti u vidu nepotpunih podataka i nepostojanja vrednosti za neke promenljive, grešaka u merenjima, retkih velikih odstupanja od srednje vrednosti i slično. Obzirom da neuronska mreža reprodukuje podatke po dinamici koja oslikava ulazne vrednosti, veoma je bitno izbegavati loše podatke za učenje. Svi ulazni podaci posmatraju se kao **matrica A[n][m]** gde je n broj merenja, a m broj promenljivih koje opisuju svako stanje. Filtriranje se svodi na nalaženje loših zapisa i njihovo uklanjanje.
 
Jedan od načina nalaženja loših zapisa se svodi na poređenje svakog merenja Xi sa srednjom vrednošću E, uz razmatranje srednjeg odstupanja. di predstavlja težinu odstupanja merenja od srednje vrednosti, ako je ovo odstupanje veće od tri celokupan zapis iz koga je razmatrana vrednost se odbacuje (ceo red matrice).

### Izjednačavanje podataka – normalizacija
Normalizacijom se svi podaci svode na zajednički opseg vrednosti, između 0 i 1. Ovaj opseg dozvoljava neuronskoj mreži da predstavlja podatke u zoni promenljivih aktivacionih funkcija poput Tangensa hiperboličkog ili Sigmoid funckije. 
 
Nmin i Nmax predstavljaju normalizovane minimalne i maksimalne vrednosti, Xmin i Xmax se odnose na početne granične vrednosti promenljive koja se normalizuje. Nakon primene normalizacije nad ulaznim skupom podataka, normalizovani skup podataka se predaje neuronskoj mreži. Neuronska mreža kojoj su predati ovako normalizovani podaci takođe na izlazu generiše "normalizovane" podatke, pa je zato neophodno primeniti denormalizaciju dobijenih izlaznih podataka, odnosno rezultata, po obrnutom principu.


## Implementacija u .NET okruženju

Za rad sa ulaznim podacima koristi se klasa Data. Podaci se čitaju iz csv fajlova, ova klasa takođe obuhvata i predprocesiranje, kao i normalizaciju ulaznih podataka. Osnovno zaduženje klase je generisanje matrice normalizovanih podataka na osnovu inicijalnog ulaznog skupa. Klasa Data sadrži atribute PATH, FILENAME, NORMALIZATIONTYPESENUM. Prvi atribut je putanja do direktorijuma u kome se nalaze ulazni fajlovi, drugi atribut je ime fajla, a treći se odnosi na tip normalizacije poput MIN_MAX.

*	DATA(STRING PATH, STRING FILENAME)
*	DOUBLE[][] RAWDATA2MATRIX(DATA R)
*	DOUBLE[][] NORMALIZE(DOUBLE[][] RAWMATRIX, NORMALIZATIONTYPESENUM NORMTYPE)
*	DOUBLE[][] DENORMALIZE(DOUBLE[][] RAWMATRIX, DOUBLE[][] MATRIXNORM, NORMALIZATIONTYPESENUM)

### Struktuiranost neuronske mreže
Ulazne promenljive neuronske mreže su:
* Padavine
*	Izlaganje suncu
*	Vlažnost vazduha
*	Brzina vetra
*	Izlaz: Srednja temperatura : predviđena srednja vrednost maksimalne i minimalne temperature

Moguće je menjati parametre neuronske mreže poput broja neurona u skrivenom sloju, mere učenja i tipa normalizacije. Promena ovih parametara utiče na krajnje rezultate i na meru odstupanja.

Klasa Neuron od ključnih atributa sadrži LISTOFWEIGHTIN i  LISTOFWEIGHTOUT. Prvi atribut je niz realnih brojeva koji predstavlja listu ulaznih težina, a srodno njemu drugi atribut predstavlja listu izlazni težina. Svaka instanca pored težina ima i izlaznu vrednost, grešku i osetljivost.
*	INITNEURON() 
*	LISTOFWEIGHTIN(ARRAYLIST<DOUBLE> LISTOFWEIGHTIN)
*	LISTOFWEIGHTOUT(ARRAYLIST<DOUBLE> LISTOFWEIGHTOUT)
*	LISTOFWEIGHTIN()
*	LISTOFWEIGHTOUT()
*	OUTPUTVALUE()
*	...
  
Klasa Layer sadrži atribute LISTOFNEURONS i NUMBEROFNEURONSINLAYER. Prvi atribut predstavlja listu svih neurona u posmatranom sloju, a drugi atribut daje uvid u broj neurona koji čine sloj.
*	LISTOFNEURONS()
*	LISTOFNEURONS(ARRAYLIST<NEURON> LISTOFNEURONS)
*	NUMBEROFNEURONSINLAYER()
*	NUMBEROFNEURONSINLAYER(INT NUMBEROFNEURONSINLAYER)
  
Klasa InputLayer nasleđuje klasu Layer i sve njene atribute uz pridodate metode. Sadrži dodatno niz neurona koji čine jedan ulazni sloj.
*	INITLAYER(INPUTLAYER INPUTLAYER)
*	PRINTLAYER(INPUTLAYER INPUTLAYER)

Klasa HiddenLayer takođe nasleđuje klasu Layer. Sadrži dodatno niz neurona koji čine jedan skriveni sloj.
*	INITLAYER(HIDDENLAYER HIDDENLAYER, ARRAYLIST<HIDDENLAYER> LISTOFHIDDENLAYER, INPUTLAYER INPUTLAYER, OUTPUTLAYER OUTPUTLAYER)
*	PRINTLAYER(ARRAYLIST<HIDDENLAYER> LISTOFHIDDENLAYER)
  
Klasa OutputLayer takođe nasleđuje klasu Layer uz male razlike  u odnosu na prethodne dve klase.
Klasa NeuralNet sadrži kroz atribute sve gore definisane slojeve. Atributi ove klase su INPUTLAYER, HIDDENLAYER, LISTOFHIDDENLAYER, OUTPUTLAYER, NUMBEROFHIDDENLAYERS. Takođe sadrži i atribute poput skupaova  za treniranje i validaciju, maksimalan broj epocha (ciklusa treninga), meru učenja, srednju grešku, listu svih mse grešaka, ciljnu grešku i tako dalje.
*	INITNET(INT NUMBEROFINPUTNEURONS, INT NUMBEROFHIDDENLAYERS, INT NUMBEROFNEURONSINHIDDENLAYER, INT NUMBEROFOUTPUTNEURONS)
*	NEURALNET TRAINNET(NEURALNET N)
*	PRINTNET()
*	GETNETOUTPUTVALUES(NEURALNET TRAINEDNET)
*	...

Klasa Training sadrži atribute EPOCHS u vidu celog broja koji služi za čuvanje ciklusa treninga, ERROR u vidu realog broja za praćenje grešaka tj. odnosa između generisanih izlaza i očekivanih i MSE takođe kao realan broj za praćenje srednje kvadratne greške.
*	ENUM TRAININGTYPESENUM {BACK_PROPAGATION, .. ;}
*	ENUM ACTIVATIONFNCENUM {STEP, LINEAR, SIGLOG,HYPERTAN;}
*	NEURALNET TRAIN(NEURALNET N)
*	TEACHNEURONSOFLAYER(INT NUMBEROFINPUTNEURONS, INT LINE, NEURALNET N, DOUBLE NETVALUE)
*	CALCNEWWEIGHT(TRAININGTYPESENUM TRAINTYPE, DOUBLE INPUTWEIGHTOLD, NEURALNET N, DOUBLE ERROR, DOUBLE TRAINSAMPLE, DOUBLE NETVALUE)
*	PUBLIC DOUBLE ACTIVATIONFNC (ACTIVATIONFNCENUM FNC, DOUBLE VALUE)
*	ACTIVATIONFNC(ACTIVATIONFNCENUM FNC, DOUBLE VALUE)
*	…

Potrebno je u folder data (\ANNetwork\bin\Debug\data) dostaviti ulazne .csv trening fajlove:
1. input.csv - bias(naklonjenost), padavine, izlaganje suncu, vlažnost vazduha i brzina vetra po uzorku
2. otput.csv - srednja dnevna temperatura za svaki od prethnodnih uzoraka

Kao i fajlove za testiranje neuronske mreže koji su identično struktuirani – stvarni podaci za proveru neuronske mreže.
input_test.csv
output_test.csv

Izborom File-Execute iz Menu bar-a se pokreće proces učenja i treniranja neuronske mreže. Nakon toga se prikazuju podaci koji uključuju izlaz neuronske mreže i stvarne podatke. Takođe je prikazana i mera greške koja na početku treniranja drastično opada i normalizuje se. Stvarni podaci su preslikani iz fajla output_test.csv dok su ulazni podaci za proveru efikasnosti neuronske mreže uzeti iz fajla input_test.csv, na osnovu ovih uzoraka su generisana predviđanja od strane neuronske mreže. Dakle, konačna mera efikasnosti neuronske mreže za predviđanje srednje temperature na osnovu ostalih ulaznih vremenskih promenljivih je na gornjem desnom grafiku. Prilikom treniranja neuronske mreže koriščeni su fajlovi input.csv i output.csv koji imaju znatno više uzoraka, gornji levi grafik se odnosi na proces treniranja neuronske mreže.
Funkcija koja inicijalizuje neuronsku mrežu i pokreće proces učenja odnosno treininga je void NeuralExe(). Unutar ove funkcije moguće je menjati parametre neuronske mreže poput broja neurona u skrivenom sloju, mere učenja i tipa normalizacije. Promena ovih parametara utiče na krajnje rezultate i na meru odstupanja. Funkcija se nalazi u Form1.cs izvornom fajlu, odakle se i poziva rokovodiocem događaja Menu bar-a.
