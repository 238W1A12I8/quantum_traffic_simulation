import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class OptimizationMethod(Enum):
    FIXED_TIMING = "Fixed Timing"
    CLASSICAL_GA = "Classical Optimization"
    QUANTUM_ANNEALING = "Quantum Optimization"

@dataclass
class TrafficMetrics:
    avg_delay: float
    throughput: float
    queue_length: float
    efficiency: float
    total_stops: int
    fuel_consumption: float

class TrafficIntersection:
    def __init__(self, num_phases: int = 4):
        self.num_phases = num_phases
        self.current_phase = 0
        self.cycle_time = 120  # seconds
        self.min_green = 10
        self.max_green = 60
        self.yellow_time = 3
        self.all_red = 2
        
        # Traffic flow parameters
        self.arrival_rates = [0.3, 0.4, 0.35, 0.25]  # vehicles per second
        self.saturation_flow = 0.5  # vehicles per second
        
        # Simulation state
        self.queues = [0] * num_phases
        self.accumulated_delay = 0
        self.vehicles_served = 0
        self.total_vehicles = 0
        
    def update_traffic(self, green_times: List[int]) -> TrafficMetrics:
        """Update traffic simulation for one cycle with given green times"""
        total_delay = 0
        total_served = 0
        max_queue = 0
        
        for phase in range(self.num_phases):
            green_time = green_times[phase]
            
            # Calculate vehicles arriving during this phase
            arrivals = np.random.poisson(self.arrival_rates[phase] * green_time)
            self.queues[phase] += arrivals
            self.total_vehicles += arrivals
            
            # Calculate vehicles that can be served
            max_departures = int(self.saturation_flow * green_time)
            actual_departures = min(self.queues[phase], max_departures)
            
            # Update queue
            self.queues[phase] -= actual_departures
            total_served += actual_departures
            self.vehicles_served += actual_departures
            
            # Calculate delay (simplified model)
            phase_delay = self.queues[phase] * green_time * 0.5
            total_delay += phase_delay
            
            # Track maximum queue
            max_queue = max(max_queue, self.queues[phase])
            
            # Accumulate delay for vehicles in queue
            self.accumulated_delay += self.queues[phase] * green_time
        
        # Calculate metrics
        avg_delay = total_delay / max(total_served, 1)
        throughput = total_served
        efficiency = total_served / sum(green_times) if sum(green_times) > 0 else 0
        fuel_consumption = total_delay * 0.01  # Simplified fuel model
        
        return TrafficMetrics(
            avg_delay=avg_delay,
            throughput=throughput,
            queue_length=max_queue,
            efficiency=efficiency,
            total_stops=int(sum(self.queues)),
            fuel_consumption=fuel_consumption
        )

class ClassicalOptimizer:
    """Genetic Algorithm for traffic signal optimization"""
    
    def __init__(self, num_phases: int = 4):
        self.num_phases = num_phases
        self.population_size = 20  # Reduced for faster execution
        self.generations = 50     # Reduced for faster execution
        self.mutation_rate = 0.1
        self.elite_size = 2
        
    def optimize(self, intersection: TrafficIntersection, current_queues: List[int]) -> List[int]:
        """Optimize green times using Genetic Algorithm"""
        population = self._initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness = [self._fitness(individual, intersection, current_queues) 
                      for individual in population]
            
            # Select parents
            parents = self._select_parents(population, fitness)
            
            # Create new generation
            new_population = self._crossover(parents)
            new_population = self._mutate(new_population)
            
            # Elitism
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            for i, idx in enumerate(elite_indices):
                new_population[i] = population[idx]
                
            population = new_population
        
        # Return best solution
        best_idx = np.argmax([self._fitness(ind, intersection, current_queues) 
                             for ind in population])
        return self._decode_solution(population[best_idx], intersection)
    
    def _initialize_population(self) -> List[List[float]]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(0, 1) for _ in range(self.num_phases)]
            population.append(individual)
        return population
    
    def _fitness(self, individual: List[float], intersection: TrafficIntersection, 
                 queues: List[int]) -> float:
        """Calculate fitness - minimize delay, maximize throughput"""
        green_times = self._decode_solution(individual, intersection)
        
        # Create a temporary intersection for simulation without affecting the real one
        temp_intersection = TrafficIntersection(intersection.num_phases)
        temp_intersection.queues = queues.copy()
        temp_intersection.arrival_rates = intersection.arrival_rates.copy()
        
        # Simulate one cycle
        metrics = temp_intersection.update_traffic(green_times)
        
        # Fitness combines multiple objectives
        fitness = (metrics.throughput * 2 - metrics.avg_delay * 0.1 - 
                  metrics.queue_length * 0.05)
        
        # Penalty for constraint violations
        total_time = sum(green_times) + intersection.yellow_time * intersection.num_phases
        if total_time > intersection.cycle_time:
            fitness -= (total_time - intersection.cycle_time) * 10
            
        return max(fitness, 0)
    
    def _decode_solution(self, individual: List[float], intersection: TrafficIntersection) -> List[int]:
        """Decode normalized values to actual green times"""
        total = sum(individual)
        if total == 0:
            return [intersection.min_green] * self.num_phases
            
        # Scale to available green time (considering yellow and all-red times)
        available_green = (intersection.cycle_time - 
                          intersection.num_phases * (intersection.yellow_time + intersection.all_red))
        
        green_times = [int((val / total) * available_green) for val in individual]
        
        # Apply min/max constraints
        for i in range(len(green_times)):
            green_times[i] = max(intersection.min_green, 
                                min(intersection.max_green, green_times[i]))
        
        return green_times
    
    def _select_parents(self, population: List[List[float]], fitness: List[float]) -> List[List[float]]:
        """Tournament selection"""
        parents = []
        for _ in range(len(population)):
            # Tournament of size 3
            contestants = random.sample(range(len(population)), 3)
            winner = max(contestants, key=lambda x: fitness[x])
            parents.append(population[winner])
        return parents
    
    def _crossover(self, parents: List[List[float]]) -> List[List[float]]:
        """Single-point crossover"""
        children = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                point = random.randint(1, self.num_phases - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
                children.extend([child1, child2])
        return children if children else parents
    
    def _mutate(self, population: List[List[float]]) -> List[List[float]]:
        """Gaussian mutation"""
        for individual in population:
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    individual[i] = max(0, min(1, individual[i] + random.gauss(0, 0.1)))
        return population

class QuantumOptimizer:
    """Quantum-inspired optimizer using simulated quantum annealing"""
    
    def __init__(self, num_phases: int = 4):
        self.num_phases = num_phases
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.iterations = 100  # Reduced for faster execution
        
    def optimize(self, intersection: TrafficIntersection, current_queues: List[int]) -> List[int]:
        """Optimize using simulated quantum annealing"""
        current_solution = self._initialize_solution()
        current_energy = self._energy_function(current_solution, intersection, current_queues)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temp = self.temperature
        
        for iteration in range(self.iterations):
            # Generate neighbor solution using quantum-inspired tunneling
            neighbor = self._quantum_tunnel(current_solution, temp)
            neighbor_energy = self._energy_function(neighbor, intersection, current_queues)
            
            # Quantum acceptance criteria
            if self._accept_solution(current_energy, neighbor_energy, temp):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if neighbor_energy < best_energy:
                    best_solution = neighbor.copy()
                    best_energy = neighbor_energy
            
            # Quantum annealing schedule
            temp *= self.cooling_rate
            
            # Quantum tunneling effect - occasional random jumps
            if iteration % 20 == 0:
                current_solution = self._quantum_tunnel(current_solution, 1.0)
                current_energy = self._energy_function(current_solution, intersection, current_queues)
        
        return self._decode_quantum_solution(best_solution, intersection)
    
    def _initialize_solution(self) -> List[float]:
        """Initialize quantum state representation"""
        return [random.uniform(-1, 1) for _ in range(self.num_phases)]
    
    def _energy_function(self, solution: List[float], intersection: TrafficIntersection,
                        queues: List[int]) -> float:
        """Quantum energy function - lower is better"""
        green_times = self._decode_quantum_solution(solution, intersection)
        
        # Create a temporary intersection for simulation
        temp_intersection = TrafficIntersection(intersection.num_phases)
        temp_intersection.queues = queues.copy()
        temp_intersection.arrival_rates = intersection.arrival_rates.copy()
        
        metrics = temp_intersection.update_traffic(green_times)
        
        # Energy combines traffic metrics with quantum penalties
        energy = (metrics.avg_delay * 2 + metrics.queue_length * 1.5 - 
                 metrics.throughput * 0.8)
        
        # Quantum constraint penalties
        total_time = sum(green_times) + intersection.yellow_time * intersection.num_phases
        if total_time > intersection.cycle_time:
            energy += (total_time - intersection.cycle_time) * 20
            
        return energy
    
    def _quantum_tunnel(self, solution: List[float], temperature: float) -> List[float]:
        """Quantum tunneling effect for escaping local minima"""
        new_solution = solution.copy()
        phase = random.randint(0, self.num_phases - 1)
        
        # Quantum superposition-inspired mutation
        tunnel_strength = temperature * random.gauss(0, 0.3)
        new_solution[phase] = max(-1, min(1, new_solution[phase] + tunnel_strength))
        
        return new_solution
    
    def _accept_solution(self, current_energy: float, new_energy: float, 
                        temperature: float) -> bool:
        """Quantum-inspired acceptance probability"""
        if new_energy < current_energy:
            return True
        
        # Quantum tunneling probability
        delta_energy = new_energy - current_energy
        tunnel_prob = np.exp(-delta_energy / (temperature + 1e-8))
        return random.random() < tunnel_prob
    
    def _decode_quantum_solution(self, quantum_state: List[float], intersection: TrafficIntersection) -> List[int]:
        """Decode quantum state to green times"""
        # Convert quantum amplitudes to probabilities
        probabilities = [abs(state) for state in quantum_state]
        total = sum(probabilities)
        
        if total == 0:
            return [intersection.min_green] * self.num_phases
            
        available_green = (intersection.cycle_time - 
                          intersection.num_phases * (intersection.yellow_time + intersection.all_red))
        
        green_times = [int((prob / total) * available_green) for prob in probabilities]
        
        # Apply constraints
        for i in range(len(green_times)):
            green_times[i] = max(intersection.min_green, 
                                min(intersection.max_green, green_times[i]))
            
        return green_times

class TrafficSimulation:
    """Main simulation class comparing all optimization methods"""
    
    def __init__(self, simulation_duration: int = 30):  # Reduced for faster testing
        self.simulation_duration = simulation_duration  # number of cycles
        self.intersection = TrafficIntersection()
        self.classical_optimizer = ClassicalOptimizer()
        self.quantum_optimizer = QuantumOptimizer()
        
        # Results storage
        self.results = []
        
    def run_fixed_timing(self) -> List[Dict]:
        """Run simulation with fixed timing plan"""
        print("🚦 Running Fixed Timing Simulation...")
        fixed_green_times = [30, 30, 30, 30]  # Equal fixed timing
        
        results = []
        for cycle in range(self.simulation_duration):
            metrics = self.intersection.update_traffic(fixed_green_times)
            
            results.append({
                'method': OptimizationMethod.FIXED_TIMING.value,
                'cycle': cycle,
                'avg_delay': metrics.avg_delay,
                'throughput': metrics.throughput,
                'queue_length': metrics.queue_length,
                'efficiency': metrics.efficiency,
                'total_stops': metrics.total_stops,
                'fuel_consumption': metrics.fuel_consumption,
                'green_times': fixed_green_times
            })
            
            # Add some traffic variation
            if cycle % 10 == 0:
                self.intersection.arrival_rates = [r * random.uniform(0.8, 1.2) 
                                                 for r in self.intersection.arrival_rates]
        
        return results
    
    def run_classical_optimization(self) -> List[Dict]:
        """Run simulation with classical genetic algorithm optimization"""
        print("🎯 Running Classical Optimization Simulation...")
        
        results = []
        for cycle in range(self.simulation_duration):
            # Get optimized green times
            start_time = time.time()
            green_times = self.classical_optimizer.optimize(
                self.intersection, self.intersection.queues.copy())
            optimization_time = time.time() - start_time
            
            metrics = self.intersection.update_traffic(green_times)
            
            results.append({
                'method': OptimizationMethod.CLASSICAL_GA.value,
                'cycle': cycle,
                'avg_delay': metrics.avg_delay,
                'throughput': metrics.throughput,
                'queue_length': metrics.queue_length,
                'efficiency': metrics.efficiency,
                'total_stops': metrics.total_stops,
                'fuel_consumption': metrics.fuel_consumption,
                'green_times': green_times,
                'optimization_time': optimization_time
            })
            
            # Dynamic traffic patterns
            if cycle % 10 == 0:
                self.intersection.arrival_rates = [r * random.uniform(0.7, 1.3) 
                                                 for r in self.intersection.arrival_rates]
            
            print(f"  Cycle {cycle+1}/{self.simulation_duration} - Delay: {metrics.avg_delay:.1f}s")
        
        return results
    
    def run_quantum_optimization(self) -> List[Dict]:
        """Run simulation with quantum-inspired optimization"""
        print("⚛️  Running Quantum Optimization Simulation...")
        
        results = []
        for cycle in range(self.simulation_duration):
            # Get optimized green times using quantum-inspired method
            start_time = time.time()
            green_times = self.quantum_optimizer.optimize(
                self.intersection, self.intersection.queues.copy())
            optimization_time = time.time() - start_time
            
            metrics = self.intersection.update_traffic(green_times)
            
            results.append({
                'method': OptimizationMethod.QUANTUM_ANNEALING.value,
                'cycle': cycle,
                'avg_delay': metrics.avg_delay,
                'throughput': metrics.throughput,
                'queue_length': metrics.queue_length,
                'efficiency': metrics.efficiency,
                'total_stops': metrics.total_stops,
                'fuel_consumption': metrics.fuel_consumption,
                'green_times': green_times,
                'optimization_time': optimization_time
            })
            
            # More aggressive traffic changes to test adaptability
            if cycle % 10 == 0:
                self.intersection.arrival_rates = [r * random.uniform(0.6, 1.4) 
                                                 for r in self.intersection.arrival_rates]
            
            print(f"  Cycle {cycle+1}/{self.simulation_duration} - Delay: {metrics.avg_delay:.1f}s")
        
        return results
    
    def run_comparative_study(self) -> pd.DataFrame:
        """Run complete comparative study"""
        print("🔬 Starting Comprehensive Traffic Optimization Study")
        print("=" * 60)
        
        all_results = []
        
        # Run Fixed Timing
        print("\n1. FIXED TIMING BASELINE")
        self.intersection = TrafficIntersection()
        fixed_results = self.run_fixed_timing()
        all_results.extend(fixed_results)
        
        # Run Classical Optimization
        print("\n2. CLASSICAL OPTIMIZATION (Genetic Algorithm)")
        self.intersection = TrafficIntersection()
        classical_results = self.run_classical_optimization()
        all_results.extend(classical_results)
        
        # Run Quantum Optimization
        print("\n3. QUANTUM OPTIMIZATION (Simulated Annealing)")
        self.intersection = TrafficIntersection()
        quantum_results = self.run_quantum_optimization()
        all_results.extend(quantum_results)
        
        # Combine all results
        results_df = pd.DataFrame(all_results)
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save simulation results to files"""
        # Create results directory
        if not os.path.exists('simulation_results'):
            os.makedirs('simulation_results')
        
        # Save CSV
        results_df.to_csv('simulation_results/traffic_simulation_results.csv', index=False)
        
        # Save summary statistics
        summary = results_df.groupby('method').agg({
            'avg_delay': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'queue_length': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'fuel_consumption': ['mean', 'std']
        }).round(3)
        
        summary.to_csv('simulation_results/summary_statistics.csv')
        
        print("✅ Results saved to 'simulation_results/' directory")
    
    def plot_real_time_comparison(self, results_df: pd.DataFrame):
        """Create real-time comparison visualization"""
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        methods = results_df['method'].unique()
        colors = {'Fixed Timing': '#FF6B6B', 
                 'Classical Optimization': '#4ECDC4', 
                 'Quantum Optimization': '#45B7D1'}
        
        # Plot 1: Average Delay Over Time
        for method in methods:
            method_data = results_df[results_df['method'] == method]
            ax1.plot(method_data['cycle'], method_data['avg_delay'], 
                    label=method, color=colors[method], linewidth=2)
        ax1.set_title('Average Delay Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Delay (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Throughput Over Time
        for method in methods:
            method_data = results_df[results_df['method'] == method]
            ax2.plot(method_data['cycle'], method_data['throughput'], 
                    label=method, color=colors[method], linewidth=2)
        ax2.set_title('Throughput Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('Vehicles per Cycle')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Queue Length Comparison
        queue_data = results_df.groupby('method')['queue_length'].mean()
        ax3.bar(queue_data.index, queue_data.values, 
               color=[colors[method] for method in queue_data.index])
        ax3.set_title('Average Queue Length Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Queue Length (vehicles)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(queue_data.values):
            ax3.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Efficiency Comparison
        efficiency_data = results_df.groupby('method')['efficiency'].mean()
        ax4.bar(efficiency_data.index, efficiency_data.values,
               color=[colors[method] for method in efficiency_data.index])
        ax4.set_title('Traffic Efficiency Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Efficiency Score')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(efficiency_data.values):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('simulation_results/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    print("🚗 TRAFFIC SIGNAL OPTIMIZATION SIMULATION")
    print("🔬 Comparative Study: Fixed vs Classical vs Quantum")
    print("=" * 60)
    
    # Initialize simulation
    simulation = TrafficSimulation(simulation_duration=20)  # Reduced for testing
    
    # Run comparative study
    start_time = time.time()
    results_df = simulation.run_comparative_study()
    end_time = time.time()
    
    print(f"\n⏱️  Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Display summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    summary = results_df.groupby('method').agg({
        'avg_delay': 'mean',
        'throughput': 'mean', 
        'queue_length': 'mean',
        'efficiency': 'mean'
    }).round(2)
    
    print(summary)
    
    # Calculate improvements
    fixed_metrics = results_df[results_df['method'] == 'Fixed Timing'].mean(numeric_only=True)
    classical_metrics = results_df[results_df['method'] == 'Classical Optimization'].mean(numeric_only=True)
    quantum_metrics = results_df[results_df['method'] == 'Quantum Optimization'].mean(numeric_only=True)
    
    classical_delay_improvement = ((fixed_metrics['avg_delay'] - classical_metrics['avg_delay']) / fixed_metrics['avg_delay']) * 100
    quantum_delay_improvement = ((fixed_metrics['avg_delay'] - quantum_metrics['avg_delay']) / fixed_metrics['avg_delay']) * 100
    quantum_advantage = quantum_delay_improvement - classical_delay_improvement
    
    print(f"\n🚀 PERFORMANCE IMPROVEMENTS vs FIXED TIMING")
    print(f"Classical Optimization: {classical_delay_improvement:+.1f}% delay reduction")
    print(f"Quantum Optimization:   {quantum_delay_improvement:+.1f}% delay reduction")
    print(f"Quantum Advantage:      {quantum_advantage:+.1f}% additional improvement")
    
    # Generate visualizations
    simulation.plot_real_time_comparison(results_df)
    
    print("\n🎉 Simulation completed successfully!")
    print("📁 Check 'simulation_results/' folder for detailed results")

if __name__ == "__main__":
    main()