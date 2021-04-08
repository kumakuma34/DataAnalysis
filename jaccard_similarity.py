doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

print(tokenized_doc1)
print(tokenized_doc2)

union = set(tokenized_doc1).union(set(tokenized_doc2))
print(union)

intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print(intersection)

print(len(intersection) / len(union))