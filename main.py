# Author : Ajay Singh
# Created on : Sun, Jul 14 2019
# Description : Entry point for Simple Question Answer Chatbot
# Program Argument(s) : None
#
# Usage :
#		$ python3 main.py

from ProcessedQuestion import ProcessedQuestion as PQ
from AnswerRetrievalModel import AnswerRetrievalModel as ARM
from models import Schema, Add_row
import re
import sys
import sqlite3

class Chat:

    def __init__(self):
        self.conn = sqlite3.connect('chatbot.db')
        Schema()
        self.main()
        
    def main(self):
        if len(sys.argv) > 1:
            print("Bot> Parameters are not needed!")
            print("Bot> Please! Rerun me using following syntax")
            print("\t\t$ python3 main.py ")
            print("Bot> Thanks! Bye")
            exit()

        print("Bot> Hey! My name is Naubot and I am ready to answer your queries.")
        print("Bot> I was designed to give you the best answers to your questions based on what is stored in my database. You can start asking your\
                        questions or just say Hi!")
        print("Bot> To quit, you can say Bye anytime you want.")

        # Greet Pattern
        greetPattern = re.compile(
            "^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$", re.IGNORECASE)

        print("Enter 1 to Set question and answers & 2 to interact with the chatbot to retrieve answers & anything else to quit")
        
        user_input = int(input("You> "))
        if user_input == 1:
            while True:
                print("Bot> Enter a question or 'quit' to exit")
                user_q = input("You> ")
                if (not len(user_q) > 0):
                    print("Bot> Enter a valid question")
                else:
                    if user_q == "quit":
                        break;
                    else:
                        print("Bot> Enter a response for this question")
                        user_a = input("You> ")
                        if (not len(user_a) > 0):
                            print("Bot> Enter a valid response for this question")
                        else:
                            p = PQ(question = user_q)
                            a = ARM(question = user_q)
                            q_type = p.determineAnswerType(question=user_q)
                            a_type = p.determineQuestionType(question=user_q)
                            tf_count = a.getTermFrequencyCount(question=user_q)

                            Add_row(user_q, user_a, q_type, a_type, tf_count )

        elif user_input == 2: 
            question=[] 
            query = """ 
                    SELECT question FROM Chat
                        """
            r = self.conn.execute(query)
            for row in r:
                question.append(row[0])
                
            arm = ARM(question = question)

            while True:
                userQuery = input("You> ")
                if(not len(userQuery) > 0):
                    print("Bot> You need to ask something")
                elif greetPattern.findall(userQuery):
                    response = "Hello!"
                elif userQuery.strip().lower() == "bye":
                    response = "Bye Bye!"
                    isActive = False
                else:
                    # Process Question
                    pq = PQ(userQuery,True,False,True)
                    # Get Response From Bot
                    response =arm.query(pq)
                print("Bot>",response)
        else :
            print("Quitting...")

if __name__ == "__main__":        # on running python app.py
    Chat()