type MessageHandler = (data: any) => void

export class WebSocketClient {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private messageHandlers: MessageHandler[] = []

  constructor(url: string) {
    this.url = url
  }

  connect() {
    this.ws = new WebSocket(this.url)

    this.ws.onopen = () => {
      console.log('WebSocket Connected')
      this.reconnectAttempts = 0
    }

    this.ws.onmessage = (event) => {
      this.handleMessage(event)
    }

    this.ws.onclose = () => {
      console.log('WebSocket Closed')
      this.reconnect()
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket Error:', error)
    }
  }

  private handleMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data)
      this.messageHandlers.forEach(handler => handler(data))
    } catch (e) {
      console.error('Parse Error:', e)
    }
  }

  private reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      setTimeout(() => this.connect(), 3000)
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }

  close() {
    if (this.ws) {
      this.ws.close()
    }
  }

  onMessage(handler: MessageHandler) {
    this.messageHandlers.push(handler)
  }

  offMessage(handler: MessageHandler) {
    this.messageHandlers = this.messageHandlers.filter(h => h !== handler)
  }
}
