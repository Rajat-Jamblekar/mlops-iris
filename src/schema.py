from pydantic import BaseModel, conlist

class IrisInput(BaseModel):
    features: conlist(float, min_items=4, max_items=4)


##this means api will expect the input like :
# {
#   "features": [5.1, 3.5, 1.4, 0.2]
# }