from pydantic import BaseModel, Field

"""
Schema fields mirror the training CSV after dropping CustomerId/Surname/Exited.
Adjust these fields if your dataset columns differ.
"""


class PredictionRequest(BaseModel):
    CreditScore: float = Field(..., description="Customer credit score")
    Geography: str = Field(..., description="Country/region")
    Gender: str = Field(..., description="Gender")
    Age: float = Field(..., description="Age")
    Tenure: float = Field(..., description="Years with the bank")
    Balance: float = Field(..., description="Account balance")
    NumOfProducts: float = Field(..., description="Number of products")
    HasCrCard: int = Field(..., description="1 if customer has a credit card else 0")
    IsActiveMember: int = Field(..., description="1 if active member else 0")
    EstimatedSalary: float = Field(..., description="Estimated salary")


class PredictionResponse(BaseModel):
    label: str
    probability: float
