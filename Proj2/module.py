class Module(object) :

    def __init__(self):
        self.out = None

    def forward ( self , * input ) :
        raise NotImplementedError

    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError

    def param (self) : 
        return []