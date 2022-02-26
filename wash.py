# A Sample class with init method  
class Person:  
      
    # init method or constructor   
    def __init__(self, name):  
        self.name = name  
      
    # Sample Method   
    def say_hi(self):  
        print('Hello, my name is', self.name)  
      
p = Person('Nikhil')  
p.say_hi()

class Truck:

    def __init__(self, name, model, id):
        self.name, self.model, self.id = name, model, id
    
    def func(self):
        print("Name is " + self.name + "\n" + "Model is " + self.model + "\n"  + self.id)

obc = Truck("Diesel", "Brown", "100")
obc.func()