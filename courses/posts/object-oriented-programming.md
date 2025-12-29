---
title: Object Oriented Programming
category: CS
semester: 2022 F
---

# Object Oriented Programming

This post summarizes Object-Oriented Programming as a **design and abstraction paradigm**,
using Java as the reference language.

## 1. Programming Foundations

OOP is built on imperative programming: **state, control flow, and procedural abstraction**.

### Variables & Types

```java
int x = 0;
double rate = 0.1;
boolean active = true;
```
- Variables represent mutable state
- Types restrict operations and memory usage

### Methods
```java
public static int add(int a, int b) {
    return a + b;
}
```
- Methods encapsulate behavior & Parameters define inputs

### Scope
```java
public static void f() {
    int x = 1;
    if (x > 0) {
        int y = 2;
    }
}
```
- Scope limits visibility & Prevents unintended side effects

---

## 2. Classes and Objects

Classes define **abstractions**; objects are their **runtime instances**.
This is the core of Object-Oriented Programming.

---

### Class Definition

```java
public class Counter {
    private int value;
}

Counter c = new Counter(); // Objects are created from classes
```

### Object

```java
public class Counter { // Internal state is hidden
    private int value;

    public Counter(int initial) { // Constructors
        value = initial;
    }

    public void increment() { // Access is controlled via methods
        value++;
    }

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value; // this refers to the current object
    }
}
``` 

### Classes as Abstract Data Types (ADT)
- State + operations define the abstraction
- Implementation details are hidden
- Clients depend on behavior, not representation

---

## 3. Interfaces and Abstraction

Interfaces define **contracts**.  
They separate **what a class does** from **how it does it**.

### Interface Definition

```java
public interface Stack {
    void push(int x);
    int pop();
    boolean isEmpty();
}
```

### Implementing an Interface
```java
public class ArrayStack implements Stack {
    private int[] data;
    private int top;

    public void push(int x) {
        data[++top] = x;
    }

    public int pop() {
        return data[top--];
    }

    public boolean isEmpty() {
        return top < 0;
    }
}
/* -------------------- */

Stack s = new ArrayStack(); // Programming to an Interface
```

- Interfaces define contracts
- Abstraction separates use from implementation

---

## 4. Exceptions, Testing, and Debugging

### Exception Handling

```java
try {
    int x = Integer.parseInt("abc");
} catch (NumberFormatException e) { // Errors are handled, not ignored
    System.out.println("Invalid input");
}
```

### Throwing Exceptions
```java
public void withdraw(int amount) { // Methods enforce preconditions
    if (amount < 0) {
        throw new IllegalArgumentException();
    }
}
```

### Testing
```java
assert add(2, 3) == 5; // Tests specify expected behavior
```

---
## 5. Inheritance and Generics


### Inheritance

```java
public class Animal {
    public void speak() {
        System.out.println("...");
    }
}

public class Dog extends Animal { // Subclasses inherit behavior
    public void speak() { // Methods can be overridden
        System.out.println("Bark");
    }
}
```

### Polymorphism
```java
Animal a = new Dog(); 
a.speak(); // Method calls are resolved at runtime
```

### super
```java
public class Dog extends Animal { // Refers to the superclass
    public Dog() { // Used for constructor chaining and method access
        super();
    }
}
```

### Generics
```java
public class Box<T> { // Type parameters abstract over data types 
    private T value; 

    public void set(T value) { 
        this.value = value;
    }

    public T get() {
        return value;
    }
}

Box<Integer> b = new Box<>(); // usuage
b.set(10);
int x = b.get();
```

---

## 6. Event-Driven Programming, Parallelism, and the JVM

### Event-Driven Programming

```java
button.addActionListener(e -> { // Control flow is driven by events
    System.out.println("Clicked");
});
```

### Parallel and Concurrency
``` java
Thread t = new Thread(() -> { // Multiple tasks may execute concurrently
    System.out.println("Running");
});
t.start(); 
```
  
Concurrency Risks
1. Race conditions
2. Deadlocks
3. Visibility issues
4. Correctness is harder than in sequential code

### The Java Virtual Machine (JVM)
1. Java code compiles to bytecode
2. Bytecode runs on the JVM, not directly on hardware
3. JVM manages memory, threads, and execution