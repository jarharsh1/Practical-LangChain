from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    spouse:str = 'Anju'
    age:Optional[int]=None
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10)
    
new_student1 = {'name':'Harsh','age':'32','email':'abc@yahoo.com','cgpa':7}
new_student2 = {'name': ['Harsh']}


student1 = Student(**new_student1)
print(student1)

student2 = Student(**new_student2)
print(student2)