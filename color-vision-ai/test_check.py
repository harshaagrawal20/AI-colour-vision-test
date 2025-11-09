import os

def getPossibleCount(totalEfficiency):
    n = len(totalEfficiency)
    ans = 0
    
    for x in range(n + 1):
        valid = True
        
        for i in range(x):
            if totalEfficiency[i] > n:
                valid = False
                break
        
        for i in range(x, n):
            if totalEfficiency[i] <= n:
                valid = False
                break
        
        if valid:
            ans += 1

    return ans

if __name__ == '__main__':
    # Test Case 1: n=2, totalEfficiency=[1,3] -> Expected: 2
    print("Test 1:", getPossibleCount([1, 3]), "Expected: 2")
    
    # Test Case 2: n=2, totalEfficiency=[3,4] -> Expected: 1
    print("Test 2:", getPossibleCount([3, 4]), "Expected: 1")
    
    # Test Case 3: n=3, totalEfficiency=[1,2,4] -> Expected: 2
    print("Test 3:", getPossibleCount([1, 2, 4]), "Expected: 2")
