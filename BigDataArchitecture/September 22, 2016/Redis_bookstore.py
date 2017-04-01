import redis
import getpass
import threading
import sys

class Listener(threading.Thread):
    def __init__(self, r, channels):
        threading.Thread.__init__(self)
        self.redis = r
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(channels)

    def work(self, item):
        print item['channel'], ":", item['data']

    def run(self):
        for item in self.pubsub.listen():
            if item['data'] == "KILL":
                self.pubsub.unsubscribe()
                print self, "unsubscribed and finished"
                break
            else:
                self.work(item)


r = redis.Redis(host='localhost',port=6379,db=0)
#Creation du compte administrateur
r.hset('users:admin','login','shopkeeper')
r.hset('users:admin','passwd','pass_admin')
#Creation du compte utilisateur
r.hset('users:user','login','client')
r.hset('users:user','passwd','pass_user')
cond = False

#Authentification
while(not cond):
    print "Welcome, please enter your login"
    log = raw_input()
    print "Enter your password"
    pa = getpass.getpass()
    if(log == r.hget("users:admin","login") and pa == r.hget("users:admin","passwd")):
        account="admin"
        cond=True
    elif(log == r.hget("users:user","login") and pa == r.hget("users:user","passwd")):
        account="user"
        cond=True
    if(not cond):
        print "Login, password combination wrong , please try again"
#La phase d'authentification a reussie
if (account == "admin"):
    while(cond):
        print "**********************"
        print "To list available books enter > list"
        print "To add a new book enter > add"
        print "To exit enter > exit"
        print "**********************"
        choix = raw_input()
        if(choix == "list"):
            output = r.keys("books:*")
            for i in range(0,len(output)):
                print str(r.hgetall(output[i])).strip("'").strip("{").strip("}")
        if("add" == choix):
            condi = False
            while(not condi):
                print "Enter book title"
                title = raw_input()
                print "Enter ISBN"
                isbn = raw_input()
                print "Enter author"
                aut=raw_input()
                print "Enter number of available copies"
                ava=raw_input()
                if (isbn == "" or aut=="" or title == "" or ava ==""):
                    print "Title, ISBN, availability and author cannot be empty fields !"
                else:
                    condi=True
            print "Enter language (can be left empty)"
            lang = raw_input()
            print "Enter publication year (can be left empty)"
            year = raw_input()
            print "Enter description (can be left empty)"
            desc = raw_input()
            ls = [title,isbn,aut,ava,lang,year,desc]
            for elem in ls:
                head=["Title","ISBN","Author","Availability","Language","Year","Description"][ls.index(elem)]
                if(elem == ""):
                    continue
                else:
                    r.hset("books:"+title,head,elem)
                    r.expire("books:"+title,5184000) #Book expires after 60 days without being borrowed
                    if head=="Author":
                        r.publish(elem,"New book '"+title+"' by '"+elem+"' is now available !")
                        #r.publish(elem,'KILL')    #Uncomment in order to give publisher closing power over listener's channel
        if("exit" == choix):
            cond=False
if (account == "user"):
    while(cond):
        print "**********************"
        print "To list available books enter > list"
        print "To subscribe to an author enter > subscribe author_name"
        print "To borrow book enter > borrow isbn_number"
        print "To return book enter > return isbn_number"
        print "To exit enter > exit"
        print "**********************"
        choix = raw_input()
        if(choix == "list"):
            output = r.keys("books:*")
            for i in range(0,len(output)):
                print str(r.hgetall(output[i])).strip("'").strip("{").strip("}")
        if("borrow" in choix):
            isb = choix.split(" ")[1]
            output = r.keys("books:*")
            condi=False
            for i in range(0,len(output)):
                if (r.hget(output[i],"ISBN") == isb):
                    if (int(r.hget(output[i],"Availability")) >= 1):
                        r.hincrby(output[i],"Availability",-1)
                        r.expire(output[i],5184000)
                        condi=True
                        print "Have a good read."
            if condi==False:
                print "This book is unavailable"
        if("return" in choix):
            isb = choix.split(" ")[1]
            output = r.keys("books:*")
            condi=False
            for i in range(0,len(output)):
                if (r.hget(output[i],"ISBN") == isb):
                    r.hincrby(output[i],"Availability",1)
                    condi=True
                    print "Thank you for returning this book."
            if condi==False:
                print "This is not one of our books !"
        if("subscribe" in choix):
            client = Listener(r, choix.split(" ")[1])
            client.start()
        if("exit" == choix):
            sys.exit()