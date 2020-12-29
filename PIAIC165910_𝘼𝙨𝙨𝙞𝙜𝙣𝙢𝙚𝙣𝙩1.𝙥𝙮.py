#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[ ]:


import numpy as np


# 2. Create a null vector of size 10 

# In[ ]:


import numpy as np
x = np.zeros(10)
print(x)


# 3. Create a vector with values ranging from 10 to 49

# In[ ]:


import numpy as np
v = np.arrange(10,49)


# 4. Find the shape of previous array in question 3

# In[ ]:


print(v.shape)


# 5. Print the type of the previous array in question 3

# In[ ]:


print(v.dtype)


# 6. Print the numpy version and the configuration
# 

# In[ ]:


import numpy as np
print(np.__version__)
print(np.show_config())


# 7.Print the dimension of the array in question 3
# 

# In[ ]:


np.shape(v)


# 8. Create a boolean array with all the True values

# In[ ]:


bool_arr = np.ones(10, dtype=bool)
print("Numpy Array: ")
print(bool_arr)


# 9. Create a two dimensional array
# 
# 
# 

# In[ ]:


import numpy as np
y = np.array([[1,2],[3,4]])


# 10. Create a three dimensional array
# 
# 

# In[ ]:


import numpy as np
r = np.array([[[1,2,3],[3,2,1],[5,6,7],[8,7,6]]])


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[ ]:


import numpy as np
u = np.arange(12, 38)
print("Original array:")
print(u)
print("Reverse array:")
u = u[::-1]
print(u)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[ ]:


a=np.zeros(10)
a[4]=1


# 13. Create a 3x3 identity matrix

# In[ ]:


import numpy as np
array_2D=np.identity(3)
print('3x3 matrix:')
print(array_2D)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[ ]:


arr = arr.astype('float64')


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[ ]:


arr_result = np.multiply(arr1, arr2)
print(arr_result)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[ ]:


comparison = arr1 == arr2
equal_arrays = comparison.all()
print(equal_arrays)


# 17. Extract all odd numbers from arr with values(0-9)

# In[ ]:


import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9])
a[a % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[ ]:


a[a%2 == 1] = -1 


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[ ]:


import numpy as np
arr = np.arrange(10)
arr[5:-1] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[ ]:


import numpy as np
x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[ ]:


import numpy as np
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
arr[1][1] = 12
arr


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[ ]:


print("Original array elements:")
print(arr3d)
x = arr3d.view()
x[arr0] = 64
print(arr3d)
print(x)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[ ]:


import numpy as np
arr = np.arrange(0,9)
arr.reshape(3,3)
arr[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[ ]:


import numpy as np
arr = np.arrange(0,9).reshape(3,3)
arr
arr[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[ ]:


import numpy as np
a = np.arange(0,9).reshape(3,3)
a
a[2][ :2]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[ ]:


import numpy as np
x = np.random.random((10,10))
print("Original Array:")
print(x) 
xmin, xmax = x.min(), x.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[ ]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
print("A: ",a)
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("B: ",b)
print("Common values between two arrays:")
print(np.intersect1d(a, b))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[ ]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print([key for key, val in enumerate(a) if val in set(b)])


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[ ]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data
data[names != "Will"]


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[ ]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data
print(data[names !="Will"])
print(data[names !="Joe"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[ ]:


import numpy as np
arr = np.random.randn(1,15).reshape(5,3)
arr


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[ ]:


import numpy as np
arr = np.random.randn(1,16).reshape(2,2,4)
arr


# 33. Swap axes of the array you created in Question 32

# In[ ]:


import numpy as np
arr = np.random.randn(1,16).reshape(2,2,4)
arr
data.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[ ]:


import numpy as np
R = np.arange(10)
R = np.sqrt(R)
np.where(R<0.5,0,R)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[ ]:


a1 = np.random.rand(12)
a2 = np.random.rand(12)
a1max = a1.max()
a2max = a2.max()
print("Maximum Values:")
print(a1max, a2max)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[ ]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names
names = set(names)
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[ ]:


import numpy as np
a = np.array([1,2,3,4,5]) b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
print(result)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[ ]:


import numpy

print("Printing Original array")
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
print (sampleArray)

print("Array after deleting column 2 on axis 1")
sampleArray = numpy.delete(sampleArray , 1, axis = 1) 
print (sampleArray)

arr = numpy.array([[10,10,10]])

print("Array after inserting column 2 on axis 1")
sampleArray = numpy.insert(sampleArray , 1, arr, axis = 1) 
print (sampleArray)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[ ]:


import numpy.matlib 
import numpy as np
x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[ ]:


import numpy as np
a = np.random.randint(20)
a
b = a.cumsum()
print(b)

