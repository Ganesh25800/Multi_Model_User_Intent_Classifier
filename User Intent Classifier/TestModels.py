import joblib

log_clasifier = joblib.load('Logistic Regression Model/First_Stage_User_Intent_Classifier_Model.pkl')
log_Vectorizer = joblib.load('Logistic Regression Model/First_Stage_User_Intent_Vectorizer.pkl')

svm_model = joblib.load('SVM Model/svm_user_intent_classifier.pkl')
svm_model_v2 = joblib.load('SVM Model/svm_user_intent_classifier_v2.pkl')



def getResponseLog(inp):
    word = log_Vectorizer.transform([inp])

    pred = log_clasifier.predict(word)[0]

    return "Command" if pred == 1 else "Question"

def getResponseSVM(inp):

    pred = svm_model.predict([inp])

    return "Command" if pred == 0 else "Question"

def getResponseSVM_v2(inp):

    pred = svm_model_v2.predict([inp])

    return "Command" if pred == 0 else "Question"


while True:

    inp = input('You Asked: ')

    if "and" in inp.lower():
        print('It is: Multiple Inputs')
        print()
        continue

    if len(inp.strip()) < 3:
        print("Input Must Contain more than 3 numbers")
        print()
        continue
    
    log = getResponseLog(inp)
    svm = getResponseSVM(inp)
    svm_v2 = getResponseSVM_v2(inp)

    print(f"Logistic Regression Predicts: {log}")
    print(f"Support Vector Machine Version 1 Predicts: {svm}")
    print(f"Support Vector Machine Version 2 Predicts: {svm_v2}")
    print()