# A Sample class with init method  
class Bird:  
      
    # init method or constructor   
    def __init__(self, name):  
        self.name = name  
      
    # Sample Method   
    def names(self):  
        print(self.name)  
      
p = Bird('American Golden-Plover')  
p.names()
