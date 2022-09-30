
import math

def generalizedMean(x,e):
    #Degenerate case
    
    if len(e)==1:
        return float(e[0])
    
    #Checking for NaN's in e
    
    for n in e:
        if n!=n:
            return float('nan')
    
    #Checking if x is a NaN
    
    if x!=x:
        return float('nan')
    
    #Analytic continuation for zero with negative exponents
    
    if x<0.00001 and 0 in e:
        return 0.0
    
    #Analytic continuation for Geometric Mean
    
    if abs(x) < 0.00001:
        return math.exp(sum([ math.log(n) for n in e])/float(len(e)))
    
    #Analytic continuation for infinite exponents
    
    if x==float('-inf'):
        return float(min(e))
    
    if x==float('inf'):
        return float(max(e))
    
    #General procedure
    
    return (float(sum([n**x for n in e]))/len(e))**(1.0/x)


def product(e):
    '''
    Product function analogous to the sum() function
    '''
    p=1
    for n in e:
        p*=n
    return p
