import sqlite3
from collections import defaultdict
from ProcessedQuestion import ProcessedQuestion

class Schema:
    def __init__(self):
        self.conn = sqlite3.connect('chatbot.db')
        self.create_chat_table()
 
    def __del__(self):
        # body of destructor
        self.conn.commit()
        self.conn.close()

    def create_chat_table(self):

        query = """
        CREATE TABLE IF NOT EXISTS "Chat" (
          id INTEGER PRIMARY KEY,
          question TEXT,
          response TEXT,
          question_type TEXT,
          answer_type TEXT,
          term_frequency_count TEXT,
          createdOn DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(query)

    def create_interactions_table(self):

        query = """
        CREATE TABLE IF NOT EXISTS "Interactions" (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          question TEXT,
          response TEXT,
          chatID TEXT,
          createdOn DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(query)

    def create_chat_type_table(self):
        query = """
            CREATE TABLE IF NOT EXISTS "Chat_type" (
            id INTEGER PRIMARY KEY,
            type TEXT,
        );
        """

    def query_chat(self):
        for r in [('What day is it today?', 'The day is Sunday'),
            ('What is your name?','My name is Alex')]:
            self.conn.execute('INSERT INTO Chat(question, response) VALUES(?,?)',r)
      
class Add_row:
    def __init__(self, user_q, user_a, q_type, a_type, tf_count):
        self.conn = sqlite3.connect('chatbot.db')

        query = f'insert into Chat ' \
                f'(question, response, question_type, answer_type, term_frequency_count) ' \
                f'values ("{user_q}","{user_a}","{q_type}","{a_type}","{tf_count}")'
        r = self.conn.execute(query)
        self.conn.commit()

class GetAnswer:
    def __init__(self):
        self.conn = sqlite3.connect('chatbot.db')
        # self.add_row(id)

    def add_row(self, id):

        answer = ""
        query = f'SELECT response FROM Chat'\
                f'WHERE id = ' + str(id)
     
        query = "SELECT response from Chat WHERE id = {} " .format(id)
        s = self.conn.execute(query)
        for row in s:
            answer = row
        return [answer[0]]
