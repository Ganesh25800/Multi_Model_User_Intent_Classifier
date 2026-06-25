import joblib

model_v1 = joblib.load('svm_user_intent_classifier.pkl')
model_v2 = joblib.load('svm_user_intent_classifier_v2.pkl')

def predict_v1(inp):
    predicted = model_v1.predict([inp])

    return "command" if predicted == 0 else "question"


def predict_v2(inp):
    predicted = model_v2.predict([inp])

    return "command" if predicted == 0 else "question"

def check_and(inp):

    if "and" in inp:
        return True
    else:
        return False

def start():

    while True:

        user_input = input("Input: ").lower()

        split_input = user_input.split()

        result = check_and(split_input)

        if result == False:
            v1_conff = max(model_v1.predict_proba([user_input])[0])
            v2_conff = max(model_v2.predict_proba([user_input])[0])

            print(f"SVM Model V1 Predicts: {predict_v1(user_input)} with confidence: {v1_conff}")
            print(f"SVM Model V2 Predicts: {predict_v2(user_input)} with confidence: {v2_conff}")

            if v1_conff > v2_conff:
                print(f"It is a {predict_v1(user_input)}")
                print()
            else:
                print(f"It is a {predict_v2(user_input)}")
                print() 
            print()
            continue

        print()
        print("It is a Multi Input")
        print()

        input_split = user_input.split('and')

        input_count = len(input_split)

        print(f"It has {input_count} inputs")

        for i in range(input_count):

            print(f"Input {i+1}: {input_split[i]}")

            v1 = predict_v1(input_split[i])
            v2 = predict_v2(input_split[i])

            v1_conf = max(model_v1.predict_proba([input_split[i]])[0])
            v2_conf = max(model_v2.predict_proba([input_split[i]])[0])
            
            if v1 == v2:
                print(f"SVM Model V1 Predicts: {v1} with confidence: {v1_conf}")
                print(f"SVM Model V2 Predicts: {v2} with confidence: {v2_conf}")
                print()
            else:
                print(f"SVM Model V1 Predicts: {v1} with confidence: {v1_conf}")
                print(f"SVM Model V2 Predicts: {v2} with confidence: {v2_conf}")
                print()
            
            if v1_conf > v2_conf:
                print(f"It is a {v1}")
                print()
            else:
                print(f"It is a {v2}")
                print()  


       
start()

    