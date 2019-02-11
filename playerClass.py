# Machine Learning Classsifier
# 2 Items to compare: Monkeys and Gorillas
# Feature: weight, height

from sklearn import tree

#Features = hieght & Weight
#hieght = centimeters0
#weight = kilograms
features = [[150,100], [170,150], [160,160], [0.45,3.9], [0.40,2.9], [0.30,4.00]]

labels = [1, 1, 1, 0, 0, 0]

#I used this to see if it actually worked to see if it is a ! or a 0;
#labels=["Gorilla", "Gorilla", "Gorilla", "Monkey", "Monkey", "Monkey"]

# decision decision decisions lets make decision
clf = tree.DecisionTreeClassifier()

#The Machine Learning part is right here
clf = clf.fit(features, labels)

#Unknowns
#should be Gorillas
print(clf.predict([[150,200]]))

#Should be Monkeys
print(clf.predict([[0.30,2.9]]))