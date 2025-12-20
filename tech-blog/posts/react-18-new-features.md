---
title: New Features in React 18
category: REACT
date: 2024.01.15
readtime: 8 min
---

# New Features in React 18

React 18 is a major update focused on concurrency features.

## Automatic Batching
```javascript
setTimeout(() => {
  setCount(c => c + 1);
  setFlag(f => !f);
}, 1000);
```

## Transitions
```javascript
import { startTransition } from 'react';

setInputValue(input);
startTransition(() => {
  setSearchQuery(input);
});
```

> "The best way to improve user experience is to prioritize updates."

## Conclusion

React 18 brings powerful new features for building responsive applications.