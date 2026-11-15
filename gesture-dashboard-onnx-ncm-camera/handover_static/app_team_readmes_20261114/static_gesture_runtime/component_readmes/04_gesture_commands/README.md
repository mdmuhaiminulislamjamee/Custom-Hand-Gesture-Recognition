# 04 - Connect gestures to product actions

## What this part does

The runtime recognizes gestures, confirms them over time, switches hand
ownership when qualified and prevents duplicate action firing. `onCommand`
receives only confirmed command events. The runtime does not operate the
product's motors, media controls or recording system itself.

## What the app team integrates

Pass a callback when creating `StaticGestureRuntimeApi`:

```dart
final runtime = StaticGestureRuntimeApi(
  classifier: StaticTfliteClassifier(model: StaticTfliteModel.float32),
  onCommand: (event) {
    // Route event.label to the app's existing action handler.
    // event.action describes the action; event.hand identifies the owner.
  },
);
```

Use `event.label` as the stable action key:

| Gesture label | Intended action |
|---|---|
| `left`, `right`, `up`, `down` | Move in that direction |
| `open_palm` | Enable / disable object tracking |
| `like` | Play / pause |
| `ok` | Start / stop recording |
| `fist` | Mute |
| `thumb_down` | Volume down |
| `peace` | Turn 90 degrees |
| `rock` | Backup |
| `stop_inverted` | Return to main position |

`no_gesture` and `dorsal` do not dispatch actions. Dispatch through `onCommand`
only: do not also execute `result.command`, a raw geometry label or a high
model score, because that would bypass confirmation or repeat the action.
The app team must connect these keys to its existing product commands and
decide what each action does in its UI/transport.
