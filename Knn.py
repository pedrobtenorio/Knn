from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
 
 
filename = 'C:\\Users\\Usuario\\Desktop\\ia\\knn\\sample.csv'
dataset = pd.read_csv(filename, header=None)
dataset.columns = ["altura","peso", "IMC"]
heightAndWeight = dataset.drop("IMC", axis=1)
heightAndWeight = heightAndWeight.values
BMI = dataset["IMC"]
BMI = BMI.values
X_train, X_test, y_train, y_test = train_test_split(heightAndWeight, BMI, test_size=0.30, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=7 )
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))
