# Genetic Algorithm

## Circuit Specifications:
Network noise = 0.1 (yielding input fidelity of 0.813).

Gate errors = 0.01: probability of completely depolarize the two qubits i and j it is acting upon

Measurement errors = 0.01: probability of a measurement erroneously report the opposite result

Operations: Controlled-not gates, controlled-phase gates, Coin-Z, Coin-X, Anti-coin-Y

Final statuses: Detected failure, True success, Undetected failure

## Circuit Evaluation:

We evaluate the purification circuit by using perturbative expansion.
### Success rate
success rate=true success+undetected failure
### Fidelity of the final purified Bell pairs
f=(true success)/(success rate)
### Entropy of distribution
S(f)=-(log⁡(f)∗f+log⁡((1-f)/3)∗(1-f))
      where (1-f)/3 is estimates of probability of getting B, C, or D states.
### Hashing yield
Hy(f)=1-S(f)

The variable that we will use to evaluate the circuit is
### yield
yield=p/N∗Hy(f)

Where p=the probability of success

N=the number of consumed raw Bell pairs as measurements of efficiency


## Algorithm: Genetic Algorithm
```
for 1:iterations:
    # Generate new generation of circuits
    # Generate child circuits
    for each pair of circuits in the old generation
        Combines random portions of parent pairs
    end
    # Generate mutated circuits
    for each circuit in old generation + child circuits
        Delete an operation
        Add an operation
        Swap two operations
        Switch between C-NOT and C-PHASE gate
    end
    new generation = old generation + child circuits + mutated circuits
    Evaluate all circuits and sort by fidelity
    Pick the best N circuits for next iteration
end
```

