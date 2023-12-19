# Performance of Krylov subspace methods 
Final project of AMSC 763
## Example
```sh
python main.py --matrix_name can_1054.mtx
```

## Running results

### CAN1054, dimension: 1054
| Method   | Time   | Iterations | Residual   |
| ---      | ---    | ---        | ---        |
| GMRES(20) | 27.271 | 5000       | 0.0066857  |
| GMRES(50) | 31.025 | 5000       | 0.00519931 |
| BiCG     | 0.562  | 5000       | 2.76319e-06|
| BiCGSTAB | 0.503  | 5000       | 0.0010962  |
| IDR(4)   | 0.500  | 5001       | 0.231829   |
| IDR(8)   | 0.661  | 5001       | 0.0253972  |

### DWT992, dimension: 992
| Method   | Time   | Iterations | Residual     |
| ---      | ---    | ---        | ---          |
| GMRES(20) | 30.400 | 5000       | 0.0013189    |
| GMRES(50) | 39.074 | 5000       | 0.000656055  |
| BiCG     | 0.120  | 518        | 5.37936e-09  |
| BiCGSTAB | 0.114  | 652        | 0.000408529  |
| IDR(4)   | 0.119  | 739        | 8.75928e-09  |
| IDR(8)   | 0.105  | 623        | 8.66428e-09  |

### IMPCOLA, dimension: 207
| Method   | Time   | Iterations | Residual   |
| ---      | ---    | ---        | ---        |
| GMRES(20) | 0.849  | 5000       | 0.949579   |
| GMRES(50) | 0.869  | 5000       | 0.868925   |
| BiCG     | 0.328  | 5000       | 17504.5    |
| BiCGSTAB | 0.023  | 306        | 21417.1    |
| IDR(4)   | 0.314  | 5001       | 2.67133e-06|
| IDR(8)   | 0.112  | 1343       | 4.76677e-07|
